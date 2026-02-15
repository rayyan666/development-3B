from app.agent.chat_controller import ChatController

controller = ChatController()

print("Offline Data Science Assistant (type 'exit' to quit)\n")

while True:
    user_input = input(">> ")

    if user_input.lower() == "exit":
        break

    response = controller.handle(user_input)
    print("\n", response, "\n")
