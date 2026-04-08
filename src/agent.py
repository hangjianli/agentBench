import subprocess
import re
from typing import List, Dict, Optional, Any
from src.inference import InferenceClient

class DockerTool:
    def __init__(self, image_name="gaia-evaluator"):
        self.image_name = image_name
    
    def run_python(self, code: str, timeout: int = 30) -> str:
        """Runs python code inside the docker container and returns the output."""
        try:
            # We pass the code to python via stdin. 
            # The Dockerfile ENTRYPOINT is ["python"]
            result = subprocess.run(
                ["docker", "run", "--rm", "-i", self.image_name],
                input=code,
                capture_output=True,
                timeout=timeout,
                text=True
            )
            output = result.stdout
            if result.stderr:
                output += "\n[stderr]\n" + result.stderr
            return output.strip() if output else "Execution finished with no output."
        except subprocess.TimeoutExpired:
            return "Execution timed out."
        except Exception as e:
            return f"Error executing code: {e}"

class Agent:
    def __init__(self, client: InferenceClient):
        self.client = client
        self.tool = DockerTool()
        self.system_prompt = (
            "You are an AI assistant designed to solve tasks. You can write Python code to be executed in a safe sandboxed environment.\n"
            "To execute code, use the following format:\n"
            "```python\n"
            "# your code here\n"
            "```\n"
            "The output of the code will be provided back to you in the next message. Use print() to output information you need.\n"
            "When you have the final answer, wrap it in <final_answer> tags. Keep your final answer concise as per the task requirements.\n"
            "Example: <final_answer>42</final_answer>"
        )

    def run(self, task: str, max_steps: int = 10) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task}
        ]
        
        final_answer_text = "Max steps reached without a final answer."
        
        for step in range(max_steps):
            response = self.client.chat_completion(messages)
            messages.append({"role": "assistant", "content": response})
            print(f"\n--- Step {step + 1} ---")
            print(f"Agent:\n{response}")
            
            if "<final_answer>" in response:
                match = re.search(r"<final_answer>(.*?)</final_answer>", response, re.DOTALL)
                if match:
                    final_answer_text = match.group(1).strip()
                else:
                    final_answer_text = response
                break
            
            code_blocks = re.findall(r"```python\n(.*?)\n```", response, re.DOTALL)
            if code_blocks:
                code_to_run = code_blocks[-1]
                print("\n[Executing Code...]")
                output = self.tool.run_python(code_to_run)
                print(f"[Output]:\n{output}")
                
                messages.append({
                    "role": "user", 
                    "content": f"Code Output:\n```\n{output}\n```\nContinue to solve the task or provide the final answer."
                })
            else:
                messages.append({
                    "role": "user",
                    "content": "Please continue your thought process, write python code to execute, or provide the <final_answer>."
                })
                
        return {
            "final_answer": final_answer_text,
            "trajectory": messages
        }

if __name__ == "__main__":
    # Smoke test for the agent
    client = InferenceClient()
    agent = Agent(client)
    task = "Write a python script to calculate the 10th Fibonacci number, print it, and then provide the final answer."
    print(f"Task: {task}")
    result = agent.run(task, max_steps=5)
    print(f"\nFinal Answer Extracted: {result['final_answer']}")
    print(f"Trajectory length: {len(result['trajectory'])}")
