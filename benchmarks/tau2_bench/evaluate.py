import os
import subprocess
import litellm
import argparse
from dotenv import load_dotenv

def evaluate_tau2(num_tasks: int = 0, domain: str = "airline"):
    # Load environment variables from the project .env
    env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    load_dotenv(env_path)
    
    # Map LM Studio env vars to Litellm/OpenAI format used by tau2-bench
    os.environ["OPENAI_API_KEY"] = os.getenv("LMSTUDIO_TOKEN", "lm-studio")
    os.environ["OPENAI_API_BASE"] = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    model_name = os.getenv("LMSTUDIO_MODEL", "google/gemma-4-26b-a4b")
    
    # LiteLLM needs the provider prefix. For custom OpenAI endpoints, 'openai/' works if API_BASE is set.
    litellm_model = f"openai/{model_name}"
    
    # Enable debugging to see raw requests/responses
    os.environ["LITELLM_LOG"] = "DEBUG"
    
    print(f"Starting Tau2-Bench evaluation with model: {litellm_model}")
    print(f"API Base: {os.environ['OPENAI_API_BASE']}")
    print(f"Domain: {domain}")
    if num_tasks > 0:
        print(f"Running {num_tasks} task(s)...")
    else:
        print("Running ALL tasks in the domain...")
    
    # Run the tau2 command
    cmd = [
        "uv", "run", "tau2", "run",
        "--domain", domain,
        "--agent-llm", litellm_model,
        "--user-llm", litellm_model,
        "--verbose-logs"
    ]
    
    if num_tasks > 0:
        cmd.extend(["--num-tasks", str(num_tasks)])
    
    try:
        project_root = os.path.join(os.path.dirname(__file__), "..", "..")
        subprocess.run(cmd, cwd=project_root, check=True, env=os.environ)
        print("\nTau2-Bench evaluation completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"\nError during Tau2-Bench evaluation: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Tau2-Bench evaluation.")
    parser.add_argument("--num-tasks", type=int, default=0, help="Number of tasks to run (0 for all tasks, default: 0)")
    parser.add_argument("--domain", type=str, default="airline", help="The domain to run the evaluation on (e.g., airline, retail, telecom, banking_knowledge)")
    args = parser.parse_args()
    
    evaluate_tau2(num_tasks=args.num_tasks, domain=args.domain)
