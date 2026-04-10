import os
from dotenv import load_dotenv
from tau2.registry import registry

load_dotenv()
env_constructor = registry.get_env_constructor("retail")
tasks_loader = registry.get_tasks_loader("retail")
tasks = tasks_loader()
task = next(t for t in tasks if t.id == "7")

def get_hash_for_actions(actions_dict_list):
    env = env_constructor()
    env.set_state(
        initialization_data=task.initial_state.initialization_data if task.initial_state else None,
        initialization_actions=task.initial_state.initialization_actions if task.initial_state else None,
        message_history=task.initial_state.message_history if task.initial_state else []
    )
    for act in actions_dict_list:
        try:
            env.make_tool_call(
                tool_name=act["name"],
                requestor="assistant",
                **act["arguments"],
            )
        except Exception as e:
            pass
    return env.get_db_hash()

golden_actions = [
    {
        "name": "exchange_delivered_order_items",
        "arguments": {
            "order_id": "#W6390527",
            "item_ids": ["8384507844"],
            "new_item_ids": ["1569765161"],
            "payment_method_id": "paypal_7644869"
        }
    }
]

gemma_actions = [
    {
        "name": "exchange_delivered_order_items",
        "arguments": {
            "order_id": "#W6390527",
            "item_ids": ["8384507844"],
            "new_item_ids": ["7453605304"],
            "payment_method_id": "paypal_7644869"
        }
    }
]

print("Retail Task 7 - Golden Hash:")
print(get_hash_for_actions(golden_actions))

print("\nRetail Task 7 - Qwen's Predicted Hash:")
print(get_hash_for_actions(golden_actions)) # Qwen matches Golden

print("\nRetail Task 7 - Gemma's Predicted Hash:")
print(get_hash_for_actions(gemma_actions))

