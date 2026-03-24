# 📊 LMDB Checkpointer Benchmark Report

> Generated on 2026-03-24 22:59:20

## Summary

| Scenario | MemorySaver | LMDBSaver | 🏆 Winner |
| :--- | ---: | ---: | :--- |
| Sequential Writes (1K) | **30,914** ops/s | 19,619 ops/s | **MemorySaver** |
| Concurrent Writes (15T×200) | **33,947** ops/s | 8,523 ops/s | **MemorySaver** |
| History Query (list 100) | **52,672** ops/s | 45,319 ops/s | **MemorySaver** |