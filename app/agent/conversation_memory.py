class ConversationMemory:

    def __init__(self):
        self.messages = []
        self.last_dataset_id = None
        self.last_model_id = None

    def add_user_message(self, content: str):
        self.messages.append({
            "role": "user",
            "content": content
        })

    def add_assistant_message(self, content: str):
        self.messages.append({
            "role": "assistant",
            "content": content
        })

    def add_tool_result(self, tool_name: str, result: dict):
        self.messages.append({
            "role": "tool",
            "name": tool_name,
            "content": str(result)
        })

        # Track state automatically
        if tool_name == "load_csv":
            self.last_dataset_id = result.get("dataset_id")

        if tool_name == "train_model":
            self.last_model_id = result.get("model_id")

    def get_last_dataset(self):
        return self.last_dataset_id

    def get_last_model(self):
        return self.last_model_id

    def build_prompt(self):
        """
        Convert message history into prompt format for Ollama.
        Include current system state.
        """
    
        prompt = ""
    
        # Inject system state
        if self.last_dataset_id:
            prompt += f"System State: Current dataset_id = {self.last_dataset_id}\n"
    
        if self.last_model_id:
            prompt += f"System State: Last trained model_id = {self.last_model_id}\n"
    
        prompt += "\n"
    
        for msg in self.messages:
            if msg["role"] == "user":
                prompt += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                prompt += f"Assistant: {msg['content']}\n"
            elif msg["role"] == "tool":
                prompt += f"Tool Result ({msg['name']}): {msg['content']}\n"
    
        prompt += "Assistant:"
        return prompt

