import os
import subprocess
import litellm
import argparse
from typing import List
from dotenv import load_dotenv

def evaluate_tau2(num_tasks: int = 0, domain: str = "airline", concurrency: int = None, task_ids: List[int] = None, auto_resume: bool = True, retrieval_config: str = None, save_to: str = None, embedding_model: str = None, task_split_name: str = None):
    # Load environment variables from the project .env
    env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    load_dotenv(env_path)

    # Map LM Studio env vars to Litellm/OpenAI format used by tau2-bench
    os.environ["OPENAI_API_KEY"] = os.getenv("LMSTUDIO_TOKEN", "lm-studio")
    os.environ["OPENAI_API_BASE"] = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")

    # For local embeddings, we often need to trick the client into using a specific model name
    if embedding_model:
        os.environ["OPENAI_MODEL_NAME"] = embedding_model
        # Some versions of the tau2 code check specifically for this if using openai_embeddings
        os.environ["EMBEDDING_MODEL"] = embedding_model

    model_name = os.getenv("LMSTUDIO_MODEL", "google/gemma-4-26b-a4b")

    # Get concurrency from argument, env, or default to 1
    max_concurrency = concurrency or int(os.getenv("TAU2_MAX_CONCURRENCY", "1"))

    # LiteLLM needs the provider prefix. For custom OpenAI endpoints, 'openai/' works if API_BASE is set.
    litellm_model = f"openai/{model_name}"

    # Enable debugging to see raw requests/responses
    os.environ["LITELLM_LOG"] = "DEBUG"

    print(f"Starting Tau2-Bench evaluation with model: {litellm_model}")
    print(f"API Base: {os.environ['OPENAI_API_BASE']}")
    print(f"Domain: {domain}")
    print(f"Concurrency: {max_concurrency}")
    if save_to:
        print(f"Save to: {save_to}")
    if retrieval_config:
        print(f"Retrieval Config: {retrieval_config}")
    if task_split_name:
        print(f"Task Split: {task_split_name}")
    if embedding_model:
        print(f"Embedding Model: {embedding_model}")
    if task_ids:
        print(f"Running task IDs: {task_ids}")
    elif num_tasks > 0:
        print(f"Running {num_tasks} task(s)...")
    else:
        print("Running ALL tasks in the domain...")

    # Run the tau2 command
    cmd = [
        "uv", "run", "tau2", "run",
        "--domain", domain,
        "--agent-llm", litellm_model,
        "--user-llm", litellm_model,
        "--verbose-logs",
        "--max-concurrency", str(max_concurrency)
    ]

    if auto_resume:
        cmd.append("--auto-resume")

    if save_to:
        cmd.extend(["--save-to", save_to])

    if task_split_name:
        cmd.extend(["--task-split-name", task_split_name])

    if retrieval_config:
        cmd.extend(["--retrieval-config", retrieval_config])

    if task_ids:
        cmd.append("--task-ids")
        cmd.extend([str(tid) for tid in task_ids])
    elif num_tasks > 0:
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
    parser.add_argument("--concurrency", type=int, default=None, help="Number of concurrent tasks (overrides TAU2_MAX_CONCURRENCY in .env)")
    parser.add_argument("--task-ids", type=str, nargs="+", default=None, help="Specific task IDs to run")
    parser.add_argument("--auto-resume", action="store_true", default=True, help="Automatically resume from existing save file (default: True)")
    parser.add_argument("--no-resume", action="store_false", dest="auto_resume", help="Disable automatic resume")
    parser.add_argument("--retrieval-config", type=str, default=None, help="Retrieval configuration for banking_knowledge (e.g., bm25, grep_only)")
    parser.add_argument("--save-to", type=str, default=None, help="The path to save the simulation results (relative to data/simulations/)")
    parser.add_argument("--embedding-model", type=str, default=None, help="The name of the embedding model to use for RAG tasks")
    parser.add_argument("--task-split-name", type=str, default=None, help="The name of the task split to use (e.g., base, small, full)")
    args = parser.parse_args()

    evaluate_tau2(
        num_tasks=args.num_tasks, 
        domain=args.domain, 
        concurrency=args.concurrency, 
        task_ids=args.task_ids, 
        auto_resume=args.auto_resume, 
        retrieval_config=args.retrieval_config, 
        save_to=args.save_to,
        embedding_model=args.embedding_model,
        task_split_name=args.task_split_name
    )

