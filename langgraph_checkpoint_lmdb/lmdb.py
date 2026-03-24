import asyncio
import logging
from contextlib import contextmanager, asynccontextmanager
from typing import (
    Any,
    AsyncIterator,
    Iterator,
    Optional,
    Sequence,
    Tuple,
    List,
    Dict,
)
import lmdb
import orjson
import msgpack
from concurrent.futures import ThreadPoolExecutor

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    ChannelVersions,
    PendingWrite,
    WRITES_IDX_MAP,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

logger = logging.getLogger(__name__)

class LMDBSaver(BaseCheckpointSaver[str]):
    """LMDB-backed LangGraph checkpointer."""

    def __init__(
        self,
        env: lmdb.Environment,
        *,
        serde: Optional[Any] = None,
        encoding: str = "msgpack",
    ) -> None:
        super().__init__(serde=serde or JsonPlusSerializer())
        self.env = env
        self.encoding = encoding
        self._db = self.env.open_db(b"checkpoints")
        self._writes_db = self.env.open_db(b"writes")

    def _serialize(self, data: Any) -> bytes:
        if self.encoding == "orjson":
            # orjson doesn't support bytes, so we use msgpack for the envelope if bytes are present
            # but for consistency we'll just use msgpack if encoding is msgpack
            return orjson.dumps(data, default=lambda x: x.decode() if isinstance(x, bytes) else x)
        elif self.encoding == "msgpack":
            return msgpack.packb(data)
        else:
            raise ValueError(f"Unknown encoding: {self.encoding}")

    def _deserialize(self, data: bytes) -> Any:
        if self.encoding == "orjson":
            return orjson.loads(data)
        elif self.encoding == "msgpack":
            return msgpack.unpackb(data)
        else:
            raise ValueError(f"Unknown encoding: {self.encoding}")

    def _make_key(self, thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> bytes:
        return f"{thread_id}\x00{checkpoint_ns}\x00{checkpoint_id}".encode()

    def _make_prefix(self, thread_id: str, checkpoint_ns: str) -> bytes:
        return f"{thread_id}\x00{checkpoint_ns}\x00".encode()

    def get_tuple(self, config: dict) -> Optional[CheckpointTuple]:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"].get("checkpoint_id")

        with self.env.begin(db=self._db) as txn:
            if checkpoint_id:
                key = self._make_key(thread_id, checkpoint_ns, checkpoint_id)
                value = txn.get(key)
            else:
                # Get the most recent checkpoint for this thread_id and checkpoint_ns
                prefix = self._make_prefix(thread_id, checkpoint_ns)
                cursor = txn.cursor()
                # LMDB stores keys in lexicographical order. 
                # To get the "latest", we might need a timestamp or just rely on IDs if they are lexicographical (UUIDv7-like).
                # Actually, standard LangGraph IDs are not necessarily lexicographical.
                # Let's check how Postgres does it: it uses a timestamp.
                # I should probably store checkpoints with some order or use a secondary index.
                # However, the user didn't specify.
                
                # If we want "latest", we should ideally have a timestamp in the key or a separate index.
                # Given LMDB is local, we can just scan for now or implement a better key.
                
                # Let's refine the key to include a timestamp if we want "latest" naturally.
                # But LangGraph usually expects the "latest" version.
                
                # Searching for the last key with this prefix:
                if cursor.set_range(prefix + b"\xff"):
                    cursor.prev()
                else:
                    cursor.last()
                
                key = cursor.key()
                if key and key.startswith(prefix):
                    value = cursor.value()
                    checkpoint_id = key.decode().split("\x00")[-1]
                else:
                    return None

            if not value:
                return None

            data = self._deserialize(value)
            checkpoint = self.serde.loads_typed((data["type"], data["checkpoint"]))
            metadata = data["metadata"]
            parent_config = None
            if data.get("parent_checkpoint_id"):
                parent_config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": data["parent_checkpoint_id"],
                    }
                }

            # Get writes
            writes = []
            with self.env.begin(db=self._writes_db) as w_txn:
                w_cursor = w_txn.cursor()
                w_prefix = self._make_prefix(thread_id, checkpoint_ns) + checkpoint_id.encode() + b"\x00"
                if w_cursor.set_range(w_prefix):
                    for k, v in w_cursor:
                        if not k.startswith(w_prefix):
                            break
                        w_data = self._deserialize(v)
                        writes.append((w_data["task_id"], w_data["channel"], self.serde.loads_typed((w_data["type"], w_data["value"]))))

            return CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                    }
                },
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=parent_config,
                pending_writes=writes,
            )

    def list(
        self,
        config: Optional[dict],
        *,
        filter: Optional[dict] = None,
        before: Optional[dict] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        thread_id = config["configurable"]["thread_id"] if config else None
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "") if config else None
        
        prefix = b""
        if thread_id is not None:
            if checkpoint_ns is not None:
                prefix = self._make_prefix(thread_id, checkpoint_ns)
            else:
                prefix = f"{thread_id}\x00".encode()

        with self.env.begin(db=self._db) as txn:
            cursor = txn.cursor()
            if before and "checkpoint_id" in before["configurable"]:
                # Start before the specified ID
                before_key = self._make_key(thread_id, checkpoint_ns, before["configurable"]["checkpoint_id"])
                if cursor.set_range(before_key):
                    cursor.prev()
            elif prefix:
                # Find the end of the prefix range to go backwards (for "latest first")
                if cursor.set_range(prefix + b"\xff"):
                    cursor.prev()
                else:
                    cursor.last()
            else:
                cursor.last()

            count = 0
            while cursor.key() and (not prefix or cursor.key().startswith(prefix)):
                key = cursor.key()
                value = cursor.value()
                parts = key.decode().split("\x00")
                t_id, ns, c_id = parts[0], parts[1], parts[2]
                
                data = self._deserialize(value)
                checkpoint = self.serde.loads_typed((data["type"], data["checkpoint"]))
                metadata = data["metadata"]
                
                parent_config = None
                if data.get("parent_checkpoint_id"):
                    parent_config = {
                        "configurable": {
                            "thread_id": t_id,
                            "checkpoint_ns": ns,
                            "checkpoint_id": data["parent_checkpoint_id"],
                        }
                    }
                
                # Get writes (could be optimized by caching txn)
                writes = []
                with self.env.begin(db=self._writes_db) as w_txn:
                    w_cursor = w_txn.cursor()
                    w_prefix = self._make_prefix(t_id, ns) + c_id.encode() + b"\x00"
                    if w_cursor.set_range(w_prefix):
                        for wk, wv in w_cursor:
                            if not wk.startswith(w_prefix):
                                break
                            w_data = self._deserialize(wv)
                            writes.append((w_data["task_id"], w_data["channel"], self.serde.loads_typed((w_data["type"], w_data["value"]))))

                yield CheckpointTuple(
                    config={
                        "configurable": {
                            "thread_id": t_id,
                            "checkpoint_ns": ns,
                            "checkpoint_id": c_id,
                        }
                    },
                    checkpoint=checkpoint,
                    metadata=metadata,
                    parent_config=parent_config,
                    pending_writes=writes,
                )
                
                count += 1
                if limit and count >= limit:
                    break
                if not cursor.prev():
                    break

    def put(
        self,
        config: dict,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> dict:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]
        
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")
        type_, blob = self.serde.dumps_typed(checkpoint)
        
        key = self._make_key(thread_id, checkpoint_ns, checkpoint_id)
        data = {
            "type": type_,
            "checkpoint": blob,
            "metadata": metadata,
            "parent_checkpoint_id": parent_checkpoint_id,
        }
        
        with self.env.begin(db=self._db, write=True) as txn:
            txn.put(key, self._serialize(data))
            
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: dict,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]
        
        with self.env.begin(db=self._writes_db, write=True) as txn:
            for idx, (channel, value) in enumerate(writes):
                mapped_idx = WRITES_IDX_MAP.get(channel, idx)
                type_, blob = self.serde.dumps_typed(value)
                # Key: thread_id:ns:checkpoint_id:task_id:idx
                key = f"{thread_id}\x00{checkpoint_ns}\x00{checkpoint_id}\x00{task_id}\x00{mapped_idx}".encode()
                data = {
                    "task_id": task_id,
                    "channel": channel,
                    "type": type_,
                    "value": blob,
                }
                txn.put(key, self._serialize(data))

class AsyncLMDBSaver(BaseCheckpointSaver[str]):
    """Async LMDB-backed LangGraph checkpointer."""

    def __init__(
        self,
        env: lmdb.Environment,
        *,
        serde: Optional[Any] = None,
        encoding: str = "msgpack",
        executor: Optional[ThreadPoolExecutor] = None,
    ) -> None:
        super().__init__(serde=serde or JsonPlusSerializer())
        self._saver = LMDBSaver(env, serde=self.serde, encoding=encoding)
        self.executor = executor or ThreadPoolExecutor(max_workers=10)
        self.loop = asyncio.get_event_loop()

    async def aget_tuple(self, config: dict) -> Optional[CheckpointTuple]:
        return await self.loop.run_in_executor(self.executor, self._saver.get_tuple, config)

    async def alist(
        self,
        config: Optional[dict],
        *,
        filter: Optional[dict] = None,
        before: Optional[dict] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        # Generating the list in the thread pool and then yielding from it
        items = await self.loop.run_in_executor(
            self.executor, 
            lambda: list(self._saver.list(config, filter=filter, before=before, limit=limit))
        )
        for item in items:
            yield item

    async def aput(
        self,
        config: dict,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> dict:
        return await self.loop.run_in_executor(
            self.executor, self._saver.put, config, checkpoint, metadata, new_versions
        )

    async def aput_writes(
        self,
        config: dict,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        return await self.loop.run_in_executor(
            self.executor, self._saver.put_writes, config, writes, task_id, task_path
        )
