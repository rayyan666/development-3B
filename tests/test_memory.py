from app.agent.conversation_memory import ConversationMemory

memory = ConversationMemory()

memory.add_user_message("Train a model.")
memory.add_assistant_message('{"tool_call": {"name": "train_model"}}')
memory.add_tool_result("train_model", {"model_id": "abc123"})

print("Last model:", memory.get_last_model())
print("\nPrompt:\n")
print(memory.build_prompt())
