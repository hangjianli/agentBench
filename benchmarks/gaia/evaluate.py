import os
import json
from src.inference import InferenceClient
from src.agent import Agent

def evaluate_gaia():
    data_path = os.path.join(os.path.dirname(__file__), "data", "validation.jsonl")
    
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Please run download_gaia.py first.")
        return

    # Load dataset
    tasks = []
    with open(data_path, "r") as f:
        for line in f:
            tasks.append(json.loads(line))

    # Filter for tasks that don't require an external file attachment for simplicity in this baseline
    text_only_tasks = [t for t in tasks if not t.get("file_name")]
    
    print(f"Total tasks: {len(tasks)}")
    print(f"Text-only tasks: {len(text_only_tasks)}")
    
    # We will test on the first 3 text-only tasks
    test_tasks = text_only_tasks[:3]
    
    # Create logs directory
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)

    client = InferenceClient()
    correct = 0
    
    for i, task in enumerate(test_tasks):
        print(f"\n========================================")
        print(f"Evaluating Task {i+1}/{len(test_tasks)} (ID: {task['task_id']})")
        print(f"Question: {task['Question']}")
        print(f"Expected Answer: {task['Final answer']}")
        print(f"========================================")
        
        agent = Agent(client)
        prompt = f"{task['Question']}\n\nProvide the final answer in <final_answer> tags."
        
        result = agent.run(prompt, max_steps=5)
        predicted_answer = result["final_answer"]
        trajectory = result["trajectory"]
        
        # Save trajectory to log file
        log_file = os.path.join(log_dir, f"{task['task_id']}.json")
        with open(log_file, "w") as lf:
            json.dump({
                "task_id": task["task_id"],
                "question": task["Question"],
                "expected_answer": task["Final answer"],
                "predicted_answer": predicted_answer,
                "trajectory": trajectory
            }, lf, indent=2)
        
        print(f"\n[Result] Predicted: {predicted_answer}")
        print(f"[Log] Trajectory saved to {log_file}")
        # In a real benchmark, a more robust evaluator (like exact match or LLM-as-a-judge) is used.
        expected = str(task["Final answer"]).strip().lower()
        predicted = str(predicted_answer).strip().lower()
        
        if expected in predicted:
            print("[Result] ✅ Correct!")
            correct += 1
        else:
            print("[Result] ❌ Incorrect.")
            
    print(f"\n========================================")
    print(f"Evaluation Complete. Score: {correct}/{len(test_tasks)} ({(correct/len(test_tasks))*100:.2f}%)")
    print(f"========================================")

if __name__ == "__main__":
    evaluate_gaia()
