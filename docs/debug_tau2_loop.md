# Tau2-Bench LLM Interaction Debug Loop

This diagram traces the flow of an LLM request through the `tau2` framework, highlighting the critical points where JSON parsing and LiteLLM interactions occur.

```mermaid
sequenceDiagram
    participant E as "evaluate.py (CLI)"
    participant R as "runner/batch.py (run_tasks)"
    participant O as "orchestrator/orchestrator.py (step)"
    participant A as "agent/llm_agent.py (generate_next_msg)"
    participant U as "utils/llm_utils.py (generate)"
    participant L as "LiteLLM / LM Studio"

    E->>R: Starts evaluation
    R->>O: orchestrator.step()
    O->>A: agent.generate_next_message(history)
    
    rect rgb(240, 240, 240)
        Note over A, U: Prepare Request
        A->>U: generate(model, tools, messages)
        U->>U: to_litellm_messages(messages)
        U->>U: build tools_schema
    end

    rect rgb(255, 255, 200)
        Note over U, L: LLM Interaction
        U->>L: litellm.completion(messages, tools, ...)
        L-->>U: ModelResponse (Raw Content/ToolCalls)
    end

    rect rgb(255, 200, 200)
        Note over U: Vulnerable JSON Parsing
        U->>U: extract_json_from_llm_response(raw_args)
        U->>U: json.loads(cleaned_args)
        Note right of U: ERROR: Expecting value: line 1 column 1
    end

    U-->>A: (Fails with Exception)
    A-->>O: (Fails with Exception)
    O-->>R: (Exception caught by run_with_retry)
    
    rect rgb(200, 255, 200)
        Note over R: Progress/Retry Loop
        R->>R: log: "Retry 1/3 for task 0..."
        R->>O: retry attempt
    end
```

### Critical Debug Points:
1.  **Request Format (`utils/llm_utils.py`):** Are the `litellm_messages` or `tools_schema` formatted in a way that confuses LM Studio?
2.  **Model Output (`L -> U`):** Does the model return a text thought *before* the tool call, or does it wrap the entire response in a non-JSON format?
3.  **JSON Parsing (`utils/llm_utils.py`):** The `json.loads` on `tool_call.function.arguments` is where the current error is thrown. This means `LiteLLM` successfully identified a tool call but the string inside `arguments` isn't a valid JSON object.
