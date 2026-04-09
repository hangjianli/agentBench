import json
import os
import argparse
from datetime import datetime

def generate_html(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    timestamp = data.get("timestamp", "N/A")
    agent_info = data.get("info", {}).get("agent_info", {})
    user_info = data.get("info", {}).get("user_info", {})
    domain = data.get("info", {}).get("environment_info", {}).get("domain_name", "N/A")
    
    simulations = data.get("simulations", [])
    total_sims = len(simulations)
    avg_reward = (
        sum((sim.get("reward_info") or {}).get("reward", 0.0) for sim in simulations)
        / total_sims
        if total_sims > 0
        else 0.0
    )

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Tau2-Bench Evaluation Report</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; line-height: 1.6; color: #24292e; max-width: 1000px; margin: 0 auto; padding: 20px; background-color: #f6f8fa; }}
            .header {{ background: #ffffff; padding: 20px; border-radius: 8px; border: 1px solid #e1e4e8; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
            .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-top: 15px; }}
            .summary-item {{ background: #f8f9fa; padding: 10px; border-radius: 4px; border-left: 4px solid #0366d6; }}
            .summary-item.metric {{ border-left-color: #28a745; }}
            .summary-item label {{ display: block; font-size: 12px; color: #586069; font-weight: bold; text-transform: uppercase; }}
            .simulation {{ background: #ffffff; padding: 25px; border-radius: 8px; border: 1px solid #e1e4e8; margin-bottom: 30px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
            .reward-badge {{ display: inline-block; padding: 4px 12px; border-radius: 20px; font-weight: bold; font-size: 14px; margin-bottom: 10px; }}
            .reward-success {{ background: #dafbe1; color: #1a7f37; border: 1px solid #abe9b3; }}
            .reward-failure {{ background: #ffebe9; color: #cf222e; border: 1px solid #ff8182; }}
            .message-list {{ margin-top: 20px; display: flex; flex-direction: column; gap: 15px; }}
            .message {{ padding: 15px; border-radius: 8px; max-width: 85%; position: relative; }}
            .message.assistant {{ background: #f1f8ff; align-self: flex-start; border: 1px solid #c8e1ff; }}
            .message.user {{ background: #ffffff; align-self: flex-end; border: 1px solid #e1e4e8; }}
            .message.tool {{ background: #fff5b1; align-self: center; width: 90%; font-family: monospace; font-size: 13px; border: 1px solid #ffe066; }}
            .message-header {{ font-weight: bold; font-size: 12px; margin-bottom: 5px; color: #586069; }}
            .tool-call {{ background: #e1e4e8; padding: 8px; border-radius: 4px; margin-top: 10px; font-family: monospace; font-size: 12px; }}
            .nl-assertions {{ background: #f6f8fa; padding: 15px; border-radius: 6px; margin-top: 20px; border: 1px solid #e1e4e8; }}
            .nl-assertion-item {{ margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px solid #eee; }}
            .nl-assertion-item:last-child {{ border-bottom: none; }}
            pre {{ white-space: pre-wrap; word-break: break-all; margin: 0; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Tau2-Bench Evaluation Report</h1>
            <div class="summary-grid">
                <div class="summary-item"><label>Timestamp</label>{timestamp}</div>
                <div class="summary-item"><label>Domain</label>{domain}</div>
                <div class="summary-item metric"><label>Avg Reward</label>{avg_reward:.4f}</div>
                <div class="summary-item metric"><label>Total Tasks</label>{total_sims}</div>
                <div class="summary-item"><label>Agent Model</label>{agent_info.get("llm", "N/A")}</div>
                <div class="summary-item"><label>User Model</label>{user_info.get("llm", "N/A")}</div>
            </div>
        </div>
    """

    for sim in simulations:
        reward_info = sim.get("reward_info") or {}
        reward = reward_info.get("reward", 0.0)
        reward_class = "reward-success" if reward >= 1.0 else "reward-failure"
        task_id = sim.get("task_id", "N/A")
        
        html += f"""
        <div class="simulation">
            <h2>Task {task_id} <span class="reward-badge {reward_class}">Reward: {reward}</span></h2>
            <div class="nl-assertions">
                <strong>NL Assertions:</strong>
        """
        
        for assertion in reward_info.get("nl_assertions") or []:
            status = "✅" if assertion.get("met") else "❌"
            html += f"""
                <div class="nl-assertion-item">
                    {status} <strong>{assertion.get('nl_assertion')}</strong><br>
                    <small style="color: #666;">{assertion.get('justification')}</small>
                </div>
            """
        
        html += '</div><div class="message-list">'
        
        for msg in sim.get("messages", []):
            role = msg.get("role")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls")
            
            html += f'<div class="message {role}">'
            html += f'<div class="message-header">{role.upper()}</div>'
            
            if content:
                html += f'<div class="content"><pre>{content}</pre></div>'
            
            if tool_calls:
                for tc in tool_calls:
                    args_data = tc.get("arguments", {})
                    args = json.dumps(args_data, indent=2)
                    html += f"""
                    <div class="tool-call">
                        <strong>Tool Call:</strong> {tc.get('name')}<br>
                        <pre>{args}</pre>
                    </div>
                    """
            html += "</div>"
            
        html += "</div></div>"

    html += """
    </body>
    </html>
    """
    
    output_path = json_path.replace('.json', '.html')
    with open(output_path, 'w') as f:
        f.write(html)
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", help="Path to the results.json file")
    args = parser.parse_args()
    
    out = generate_html(args.json_path)
    print(f"HTML report generated: {out}")
