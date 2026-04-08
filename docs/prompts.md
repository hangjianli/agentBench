# Agent Prompting Structure

This document outlines the current prompting strategy used in the `agentBench` ReAct loop.

## 1. System Prompt
The system prompt defines the agent's identity and its ability to use tools (specifically the Docker-based Python environment).

**Reference:** `src/agent.py`

```text
You are an AI assistant designed to solve tasks. You can write Python code to be executed in a safe sandboxed environment.
To execute code, use the following format:
```python
# your code here
```
The output of the code will be provided back to you in the next message. Use print() to output information you need.
When you have the final answer, wrap it in <final_answer> tags. Keep your final answer concise as per the task requirements.
Example: <final_answer>42</final_answer>
```

## 2. Initial Task Prompt
When a new task is received (e.g., from the GAIA benchmark), it is appended to the system prompt as a `user` message.

```text
[Task/Question from Dataset]

Provide the final answer in <final_answer> tags.
```

## 3. History & Tool Interaction
The conversation history maintains the reasoning and acting chain. Tool outputs are injected as `user` messages.

### Assistant Turn (Reasoning + Action)
```text
I need to calculate the value of X. I will use Python for this.
```python
x = 10 * 5
print(x)
```
```

### User Turn (Observation)
```text
Code Output:
```
50
```
Continue to solve the task or provide the final answer.
```

## 5. Concrete Example: GAIA Task Trace

**Task ID:** `17b5a6a3-bc87-42e8-b0fb-6ab0781ef2cc`

### Task Input
> I’m researching species that became invasive after people who kept them as pets released them. There’s a certain species of fish that was popularized as a pet by being the main character of the movie Finding Nemo. According to the USGS, where was this fish found as a nonnative species, before the year 2020? I need the answer formatted as the five-digit zip codes of the places the species was found, separated by commas if there is more than one place.

### Step 1: Agent Reasoning & Action
The agent identifies the fish and decides how to find the USGS data.

**Assistant:**
```text
The fish popularized by Finding Nemo is the Ocellaris clownfish (Amphiprion ocellaris). I need to find its nonnative distribution in the USGS Nonindigenous Aquatic Species (NAS) database. I will write a script to simulate searching or processing known data points if available, or identifying the location.
```python
# In a real scenario with internet access, the agent might use a search tool.
# Here, it might attempt to use python to fetch data if an API was available.
print("Searching USGS NAS for Amphiprion ocellaris...")
# ... agent simulates data retrieval ...
print("Found 1 specimen: Tarpon Springs, Florida (Fred Howard Park) in 2011.")
```
```

### Step 2: Observation (Tool Output)
**User:**
```text
Code Output:
```
Searching USGS NAS for Amphiprion ocellaris...
Found 1 specimen: Tarpon Springs, Florida (Fred Howard Park) in 2011.
```
Continue to solve the task or provide the final answer.
```

### Step 3: Final Answer
**Assistant:**
```text
The Ocellaris clownfish was found in Fred Howard Park, Tarpon Springs, FL. The zip code for this location is 34689.
<final_answer>34689</final_answer>
```
