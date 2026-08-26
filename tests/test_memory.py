from src.memory import ConversationMemory


def test_memory_initializes_empty():
    memory = ConversationMemory(max_history_messages=4)

    assert memory.get_messages() == []


def test_memory_stores_user_and_ai_messages():
    memory = ConversationMemory(max_history_messages=4)

    memory.add_user_message("What is the annual leave policy?")
    memory.add_ai_message("Employees receive 20 working days.")

    messages = memory.get_messages()

    assert len(messages) == 2
    assert messages[0].content == "What is the annual leave policy?"
    assert messages[1].content == "Employees receive 20 working days."


def test_memory_respects_max_history():
    memory = ConversationMemory(max_history_messages=4)

    memory.add_user_message("Question 1")
    memory.add_ai_message("Answer 1")
    memory.add_user_message("Question 2")
    memory.add_ai_message("Answer 2")
    memory.add_user_message("Question 3")
    memory.add_ai_message("Answer 3")

    messages = memory.get_messages()

    assert len(messages) == 4


def test_memory_clear():
    memory = ConversationMemory(max_history_messages=4)

    memory.add_user_message("Test question")
    memory.add_ai_message("Test answer")

    memory.clear()

    assert memory.get_messages() == []