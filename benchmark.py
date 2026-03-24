import time
import uuid
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
import lmdb

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata, empty_checkpoint
from langgraph_checkpoint_lmdb import LMDBSaver

# ─── Helpers ──────────────────────────────────────────────────────

def make_checkpoint_data():
    checkpoint = empty_checkpoint()
    metadata = CheckpointMetadata(source="input", step=0, run_id="bench", parents={})
    return checkpoint, metadata

def bar_chart(label_values: list[tuple[str, float]], unit="ops/s", width=40):
    """Generates a Unicode bar chart."""
    if not label_values:
        return ""
    max_val = max(v for _, v in label_values)
    lines = []
    for label, val in label_values:
        bar_len = int((val / max_val) * width) if max_val > 0 else 0
        bar = "█" * bar_len
        lines.append(f"  {label:<15} {bar} {val:,.0f} {unit}")
    return "\n".join(lines)

# ─── Benchmark Scenarios ─────────────────────────────────────────

def bench_sequential_writes(saver, n=1000):
    cp, meta = make_checkpoint_data()
    tid = str(uuid.uuid4())
    start = time.time()
    for _ in range(n):
        cp["id"] = str(uuid.uuid4())
        cfg = {"configurable": {"thread_id": tid, "checkpoint_ns": "", "checkpoint_id": cp["id"]}}
        saver.put(cfg, cp, meta, {})
    return time.time() - start

def bench_concurrent_writes(saver, threads=15, total=3000):
    per_thread = total // threads
    cp, meta = make_checkpoint_data()

    def worker():
        tid = f"c_{uuid.uuid4()}"
        for _ in range(per_thread):
            c = cp.copy()
            c["id"] = str(uuid.uuid4())
            cfg = {"configurable": {"thread_id": tid, "checkpoint_ns": "", "checkpoint_id": c["id"]}}
            saver.put(cfg, c, meta, {})

    start = time.time()
    with ThreadPoolExecutor(max_workers=threads) as ex:
        futs = [ex.submit(worker) for _ in range(threads)]
        for f in futs:
            f.result()
    return time.time() - start

def bench_history_query(saver):
    cp, meta = make_checkpoint_data()
    tid = f"hist_{uuid.uuid4()}"
    for _ in range(500):
        cp["id"] = str(uuid.uuid4())
        cfg = {"configurable": {"thread_id": tid, "checkpoint_ns": "", "checkpoint_id": cp["id"]}}
        saver.put(cfg, cp, meta, {})

    cfg = {"configurable": {"thread_id": tid, "checkpoint_ns": ""}}
    start = time.time()
    count = sum(1 for _ in saver.list(cfg, limit=100))
    return time.time() - start

SCENARIOS = [
    ("Sequential Writes (1K)", bench_sequential_writes, {"n": 1000}, 1000),
    ("Concurrent Writes (15T×200)", bench_concurrent_writes, {"threads": 15, "total": 3000}, 3000),
    ("History Query (list 100)", bench_history_query, {}, 100),
]

def create_saver(name):
    if name == "MemorySaver":
        return MemorySaver(), lambda: None
    elif name == "LMDBSaver":
        tmpdir = tempfile.mkdtemp()
        env = lmdb.open(tmpdir, max_dbs=10, map_size=1024*1024*1024)
        saver = LMDBSaver(env)
        def cleanup():
            env.close()
            shutil.rmtree(tmpdir)
        return saver, cleanup

BACKENDS = ["MemorySaver", "LMDBSaver"]

def run_benchmarks():
    results = {}
    for scenario_name, bench_fn, kwargs, op_count in SCENARIOS:
        results[scenario_name] = {}
        print(f"\n{'='*60}\n  {scenario_name}\n{'='*60}")
        for backend in BACKENDS:
            print(f"  [{backend}] Running...", end="", flush=True)
            saver, cleanup = create_saver(backend)
            try:
                dur = bench_fn(saver, **kwargs)
                ops = op_count / dur if dur > 0 else float("inf")
                results[scenario_name][backend] = (dur, ops)
                print(f" {dur:.3f}s ({ops:,.0f} ops/s)")
            finally:
                cleanup()

    md = ["# 📊 LMDB Checkpointer Benchmark Report\n"]
    md.append(f"> Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    md.append("## Summary\n")
    md.append("| Scenario | " + " | ".join(BACKENDS) + " | 🏆 Winner |")
    md.append("| :--- | " + " | ".join(["---:" for _ in BACKENDS]) + " | :--- |")
    for scenario_name, _, _, _ in SCENARIOS:
        row = []
        best_ops = max(results[scenario_name][b][1] for b in BACKENDS)
        winner = next(b for b in BACKENDS if results[scenario_name][b][1] == best_ops)
        for b in BACKENDS:
            ops = results[scenario_name][b][1]
            row.append(f"**{ops:,.0f}** ops/s" if ops == best_ops else f"{ops:,.0f} ops/s")
        md.append(f"| {scenario_name} | " + " | ".join(row) + f" | **{winner}** |")
    
    print("\n" + "\n".join(md))
    with open("benchmark_results.md", "w") as f:
        f.write("\n".join(md))

if __name__ == "__main__":
    run_benchmarks()
