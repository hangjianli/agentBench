# agentBench

A framework for reproducing benchmark scores for local LLMs on agentic tasks. Currently focused on the **GAIA** (General AI Assistants) benchmark.

## Agentic Loop Design

The following diagram illustrates the **ReAct (Reasoning and Acting)** loop implemented in this project:

```mermaid
sequenceDiagram
    participant U as "User/Evaluator"
    participant A as "Agent (src/agent.py)"
    participant LLM as "InferenceClient (LLM)"
    participant D as "DockerTool (Sandbox)"

    U->>A: Task (Question)
    loop until <final_answer> or max_steps
        A->>LLM: System Prompt + History + Task
        LLM-->>A: Thought + Action (```python code```)
        
        alt Code Block Detected
            A->>D: run_python(code)
            Note over D: Executes in gaia-evaluator container
            D-->>A: stdout / stderr
            A->>A: Append output to history
        else No Code Block
            A->>A: Prompt Agent to continue or finish
        end

        A->>A: Check for <final_answer>
        alt Final Answer Found
            A->>U: Extracted Final Answer + Trajectory
        end
    end
    
    Note over A,U: Log Trajectory to benchmarks/gaia/logs/
```

## Project Structure

- `src/`: Core implementation.
  - `inference.py`: Connection to local LLM backends (LM Studio).
  - `agent.py`: ReAct loop and tool definitions.
- `benchmarks/gaia/`: GAIA-specific setup.
  - `data/`: Downloaded benchmark datasets.
  - `evaluator/`: Docker configuration for sandboxed execution.
  - `evaluate.py`: Evaluation script with trajectory logging.
  - `logs/`: Trajectory logs for each task.

## Usage

1. **Setup Environment:**
   ```bash
   uv sync
   ```
2. **Download Benchmark Data:**
   ```bash
   uv run python benchmarks/gaia/download_gaia.py
   ```
3. **Run Evaluation:**
   ```bash
   uv run python -m benchmarks.gaia.evaluate
   ```
