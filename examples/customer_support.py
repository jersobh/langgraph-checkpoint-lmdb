import os
import lmdb
import uuid
from typing import TypedDict, Annotated, List, Union
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.lmdb import LMDBSaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import operator

# 1. Define the State
class SupportState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    user_id: str
    issue_resolved: bool

# 2. Define the Nodes
def support_bot(state: SupportState):
    last_message = state["messages"][-1].content.lower()
    if "help" in last_message:
        response = AIMessage(content="Sure! I can help with your order. What's your order ID?")
    elif "order" in last_message:
        response = AIMessage(content="I've found your order. It's on its way!")
        return {"messages": [response], "issue_resolved": True}
    else:
        response = AIMessage(content="How can I assist you today?")
    return {"messages": [response]}

# 3. Build the Graph
workflow = StateGraph(SupportState)
workflow.add_node("agent", support_bot)
workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

# 4. Setup LMDB Persistence
DB_PATH = "./customer_support_db"
if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)

env = lmdb.open(DB_PATH, map_size=10 * 1024 * 1024, max_dbs=1)
saver = LMDBSaver(env)

# 5. Compile and Run
app = workflow.compile(checkpointer=saver)

def simulate_conversation():
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"--- New Conversation (Thread: {thread_id}) ---")
    
    # User says hello
    input_state = {"messages": [HumanMessage(content="Hello")], "user_id": "user_123", "issue_resolved": False}
    for event in app.stream(input_state, config):
        for value in event.values():
            print(f"Bot: {value['messages'][-1].content}")

    # User asks for help
    input_state = {"messages": [HumanMessage(content="I need help")]}
    for event in app.stream(input_state, config):
        for value in event.values():
            print(f"Bot: {value['messages'][-1].content}")

    # Check state persistence - simulate a restart or new session
    print("\n--- Resuming Conversation ---")
    state = app.get_state(config)
    print(f"Last Bot Message in State: {state.values['messages'][-1].content}")

if __name__ == "__main__":
    simulate_conversation()
