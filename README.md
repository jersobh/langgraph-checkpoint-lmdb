# LangGraph LMDB Checkpointer

A high-performance, local checkpoint saver for LangGraph using LMDB (Lightning Memory-Mapped Database).

## 🚀 Why LMDB?

LMDB is a transactional, memory-mapped key-value store. It is **incredibly fast** because it uses the operating system's memory-map (`mmap`) to provide zero-copy reads and highly efficient writes. For LangGraph, where state persistence can become a bottleneck during complex agentic loops, LMDB offers a near-memory speed with full persistence.

### 🏢 When to Use?
- **High-Performance Local Agents**: When running agents on a single machine or edge device where network latency to a database (like Postgres) is unacceptable.
- **Embedded Applications**: Desktop or mobile apps that need a self-contained, lightweight, and zero-configuration database.
- **Development & Prototyping**: Fast startup and easy portability of checkpoints during the R&D phase.

### 🏆 Where it Excels
- **Read Latency**: Since it maps the database file directly into memory, reading a checkpoint is essentially a memory access.
- **Reliability (ACID)**: Fully ACID compliant with a crash-proof design. If the power fails, your checkpoints stay consistent.
- **Multi-Process/Multi-Thread**: Highly concurrent reads without blocking, making it perfect for multi-agent workflows.

## Features

- **Blazing Fast Local Storage**: Optimized for high-frequency writes and low-latency state retrieval.
- **Binary Key Strategy**: Uses a compact binary key layout (`thread_id\x00checkpoint_ns\x00checkpoint_id`) for efficient multi-index prefix scanning.
- **Flexible Serialization**: Supports both `msgpack` (default) and `orjson` for high-speed state encoding.
- **Async & Sync Support**: Provides both thread-safe synchronous (`LMDBSaver`) and non-blocking asynchronous (`AsyncLMDBSaver`) implementations.

## Installation

```bash
pip install langgraph-checkpoint-lmdb
```

## Quick Start

```python
import lmdb
from langgraph_checkpoint_lmdb import LMDBSaver
from langgraph.graph import StateGraph

# Initialize LMDB environment
env = lmdb.open("./checkpoints", max_dbs=10)
saver = LMDBSaver(env)

# Use in your LangGraph as a checkpointer
graph = builder.compile(checkpointer=saver)
```

### Real-World Examples
- **Customer Support Bot**: See [examples/customer_support.py](examples/customer_support.py) for a complete implementation of a multi-turn support agent with state persistence.
- **Interactive Chatbot**: See [examples/chatbot.py](examples/chatbot.py) for a simple interactive CLI chatbot that remembers your name across sessions.

## Performance

Benchmarks conducted on local hardware comparing `LMDBSaver` with the default `MemorySaver`.

| Scenario | MemorySaver | LMDBSaver | 🏆 Winner |
| :--- | ---: | ---: | :--- |
| Sequential Writes (1K) | 30,914 ops/s | 19,619 ops/s | MemorySaver |
| Concurrent Writes (15T×200) | 33,947 ops/s | 8,523 ops/s | MemorySaver |
| History Query (list 100) | 52,672 ops/s | 45,319 ops/s | MemorySaver |

> [!TIP]
> `LMDBSaver` handles persistent storage while maintaining performance within the same magnitude as in-memory storage, making it ideal for edge devices and high-load local agents.

## Development

### Run Tests
```bash
pytest
```

### Run Benchmarks
```bash
python benchmark.py
```

## License

MIT License. See [LICENSE](LICENSE) for details.
