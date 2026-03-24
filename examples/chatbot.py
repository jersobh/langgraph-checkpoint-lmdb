import os
import lmdb
import uuid
import sys
from typing import TypedDict, Annotated, List, Union
from langgraph.graph import StateGraph, START, END
from langgraph_checkpoint_lmdb import LMDBSaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import operator

# 1. Define the State
class ChatState(TypedDict):
    """The state of the chatbot."""
    messages: Annotated[List[BaseMessage], operator.add]
    user_name: str

# 2. Define the Nodes
def chatbot(state: ChatState):
    """The chatbot node that logic resides in."""
    messages = state["messages"]
    last_message = messages[-1].content.lower()
    user_name = state.get("user_name", "friend")

    if "my name is" in last_message:
        name = last_message.split("my name is")[-1].strip()
        response = AIMessage(content=f"Nice to meet you, {name}! I'll remember that.")
        return {"messages": [response], "user_name": name}
    
    if "who am i" in last_message or "what is my name" in last_message:
        if user_name != "friend":
            response = AIMessage(content=f"You are {user_name}! We've met before.")
        else:
            response = AIMessage(content="I don't know your name yet. What is it?")
        return {"messages": [response]}

    response = AIMessage(content=f"Hello {user_name}! I'm a persistent chatbot powered by LMDB. How can I help you today?")
    return {"messages": [response]}

# 3. Build the Graph
workflow = StateGraph(ChatState)
workflow.add_node("agent", chatbot)
workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

# 4. Setup LMDB Persistence
DB_PATH = "./chatbot_persistence_db"
if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)

# Open LMDB environment
# map_size should be large enough for your expected data
# max_dbs must be at least 2 for checkpoints and writes
env = lmdb.open(DB_PATH, map_size=10 * 1024 * 1024, max_dbs=10)
saver = LMDBSaver(env)

# 5. Compile the app
app = workflow.compile(checkpointer=saver)

def run_chat_loop():
    print("=== Interactive LMDB Chatbot ===")
    print("Type 'exit' to quit. Your state is saved automatically to LMDB.")
    
    # Try to reuse the same thread ID for persistence demonstration
    # In a real app, this would be tied to a user session
    thread_id = "demo-thread-interactive"
    config = {"configurable": {"thread_id": thread_id}}

    # Check if we have existing state
    existing_state = app.get_state(config)
    if existing_state.values:
        user_name = existing_state.values.get("user_name", "friend")
        print(f"\n[System] Found existing session for {user_name}. Resuming...")
        last_msg = existing_state.values["messages"][-1].content
        print(f"Bot: {last_msg}")
    else:
        print("\n[System] Starting a fresh session.")

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Bot: Goodbye! Your state has been saved.")
                break
            
            if not user_input:
                continue

            input_state = {"messages": [HumanMessage(content=user_input)]}
            
            # Stream the response
            for event in app.stream(input_state, config):
                for value in event.values():
                    print(f"Bot: {value['messages'][-1].content}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    run_chat_loop()
