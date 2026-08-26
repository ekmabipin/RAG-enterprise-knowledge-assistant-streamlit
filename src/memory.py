from langchain_core.messages import HumanMessage, AIMessage


class ConversationMemory:
    def __init__(self, max_history_messages: int = 10):
        self.max_history_messages = max_history_messages
        self.messages = []

    def add_user_message(self, message: str):
        self.messages.append(
            HumanMessage(content=message)
        )
        self._trim_history()

    def add_ai_message(self, message: str):
        self.messages.append(
            AIMessage(content=message)
        )
        self._trim_history()

    def get_messages(self):
        return self.messages.copy()

    def clear(self):
        self.messages = []

    def _trim_history(self):
        if len(self.messages) > self.max_history_messages:
            self.messages = self.messages[
                -self.max_history_messages:


    
            ]

if __name__ == "__main__":
    memory = ConversationMemory(max_history_messages=4)

    memory.add_user_message("What is the annual leave policy?")
    memory.add_ai_message("Employees receive 20 working days of annual leave.")
    memory.add_user_message("What about carry-forward?")
    memory.add_ai_message("Employees may carry forward up to 10 unused days.")

    print("\nConversation history")
    print("=" * 60)

    for message in memory.get_messages():
        print(
            f"{message.__class__.__name__}: "
            f"{message.content}"
        )

    print("\nTesting history limit")
    print("=" * 60)

    memory.add_user_message("Can sick leave be carried forward?")

    for message in memory.get_messages():
        print(
            f"{message.__class__.__name__}: "
            f"{message.content}"
        )

    print("\nTesting clear")
    print("=" * 60)

    memory.clear()

    print(f"Messages after clear: {len(memory.get_messages())}")