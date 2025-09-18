"""Initializes graph and interface"""

from uuid import uuid4

from workflow import initialize_graph, stream

if __name__ == "__main__":
    workflow = initialize_graph()
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "stop", "q"]:
            print("Goodbye!")
            break
        stream(workflow, user_input, str(uuid4()))
