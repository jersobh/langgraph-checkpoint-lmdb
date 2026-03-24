import os
import shutil
import tempfile
import pytest
import lmdb
from langgraph_checkpoint_lmdb import LMDBSaver

@pytest.fixture
def lmdb_env():
    tmpdir = tempfile.mkdtemp()
    env = lmdb.open(tmpdir, max_dbs=10)
    yield env
    env.close()
    shutil.rmtree(tmpdir)

def test_lmdb_saver_basic(lmdb_env):
    saver = LMDBSaver(lmdb_env)
    
    config = {"configurable": {"thread_id": "thread-1"}}
    checkpoint = {
        "v": 1,
        "id": "cp-1",
        "ts": "2024-03-24T00:00:00Z",
        "channel_values": {"a": 1},
        "channel_versions": {"a": 1},
        "versions_seen": {},
        "pending_sends": [],
    }
    metadata = {"source": "input", "step": 1, "writes": {}, "parents": {}}
    
    # Put checkpoint
    saver.put(config, checkpoint, metadata, {})
    
    # Get checkpoint
    cp_tuple = saver.get_tuple(config)
    assert cp_tuple is not None
    assert cp_tuple.checkpoint["id"] == "cp-1"
    assert cp_tuple.metadata == metadata
    assert cp_tuple.config["configurable"]["checkpoint_id"] == "cp-1"

def test_lmdb_saver_list(lmdb_env):
    saver = LMDBSaver(lmdb_env)
    config = {"configurable": {"thread_id": "thread-1"}}
    
    for i in range(5):
        cp = {
            "v": 1,
            "id": f"cp-{i}",
            "ts": f"2024-03-24T00:00:0{i}Z",
            "channel_values": {"a": i},
            "channel_versions": {"a": i},
            "versions_seen": {},
            "pending_sends": [],
        }
        meta = {"step": i}
        saver.put(config, cp, meta, {})
    
    # List all
    checkpoints = list(saver.list(config))
    assert len(checkpoints) == 5
    # Should be in reverse order (latest first)
    assert checkpoints[0].checkpoint["id"] == "cp-4"
    assert checkpoints[-1].checkpoint["id"] == "cp-0"

def test_lmdb_saver_writes(lmdb_env):
    saver = LMDBSaver(lmdb_env)
    config = {"configurable": {"thread_id": "thread-1", "checkpoint_id": "cp-1"}}
    
    writes = [("channel-1", "value-1"), ("channel-2", "value-2")]
    saver.put_writes(config, writes, "task-1")
    
    # Get tuple should include writes
    # First we need the checkpoint
    cp = {
        "v": 1,
        "id": "cp-1",
        "ts": "2024-03-24T00:00:00Z",
        "channel_values": {},
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
    }
    saver.put(config, cp, {}, {})
    
    cp_tuple = saver.get_tuple(config)
    assert len(cp_tuple.pending_writes) == 2
    assert cp_tuple.pending_writes[0][1] == "channel-1"
    assert cp_tuple.pending_writes[0][2] == "value-1"

@pytest.mark.asyncio
async def test_async_lmdb_saver(lmdb_env):
    from langgraph_checkpoint_lmdb import AsyncLMDBSaver
    saver = AsyncLMDBSaver(lmdb_env)
    
    config = {"configurable": {"thread_id": "thread-async"}}
    checkpoint = {
        "v": 1,
        "id": "cp-async",
        "ts": "2024-03-24T00:00:00Z",
        "channel_values": {"a": 100},
        "channel_versions": {"a": 1},
        "versions_seen": {},
        "pending_sends": [],
    }
    
    await saver.aput(config, checkpoint, {}, {})
    
    cp_tuple = await saver.aget_tuple(config)
    assert cp_tuple is not None
    assert cp_tuple.checkpoint["id"] == "cp-async"
    
    checkpoints = []
    async for cp in saver.alist(config):
        checkpoints.append(cp)
    assert len(checkpoints) == 1
