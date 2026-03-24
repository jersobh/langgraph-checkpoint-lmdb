import os
import shutil
import tempfile
import lmdb
from langgraph_checkpoint_lmdb import LMDBSaver
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
import operator

# Define state
class State(TypedDict):
    count: Annotated[int, operator.add]

# Define nodes
def increment(state: State):
    return {"count": 1}

# Create graph
builder = StateGraph(State)
builder.add_node("increment", increment)
builder.add_edge(START, "increment")
builder.add_edge("increment", END)

# Setup LMDB
tmpdir = tempfile.mkdtemp()
env = lmdb.open(tmpdir, max_dbs=10)
saver = LMDBSaver(env)

# Compile graph with saver
graph = builder.compile(checkpointer=saver)

# Run graph
config = {"configurable": {"thread_id": "example-thread"}}
print("First run:")
result = graph.invoke({"count": 0}, config)
print(f"Result: {result}")

print("\nSecond run (restoring state):")
result = graph.invoke({"count": 0}, config)
print(f"Result: {result}")

# Clean up
env.close()
shutil.rmtree(tmpdir)
