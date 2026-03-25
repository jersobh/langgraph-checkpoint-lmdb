# 📊 LMDB Checkpointer Benchmark Report

> Generated on 2026-03-25 03:29:06

## Summary

| Scenario | MemorySaver | LMDBSaver | 🏆 Winner |
| :--- | ---: | ---: | :--- |
| Sequential Writes (1K) | **44,930** ops/s | 15,461 ops/s | **MemorySaver** |
| Concurrent Writes (15T×200) | **31,776** ops/s | 8,205 ops/s | **MemorySaver** |
| History Query (list 100) | 38,732 ops/s | **46,692** ops/s | **LMDBSaver** |