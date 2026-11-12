import asyncio

from grasp_demo.store import RunStore, event_stream


def test_persist_history_and_replay(tmp_path):
    store = RunStore(tmp_path)
    run = store.create({"example": {"question": "Q"}})
    run.publish({"column": "rag_7b", "type": "answer", "data": {"text": "A"}, "t": 0.2})
    store.persist(run)
    items = store.history()
    assert len(items) == 1
    assert store.replay(items[0]["file"])[0]["data"]["text"] == "A"


def test_stream_does_not_duplicate_events_present_before_connection(tmp_path):
    store = RunStore(tmp_path)
    run = store.create({})
    run.publish({"type": "answer", "data": {}, "t": 0.0})
    run.close()

    async def collect():
        return [chunk async for chunk in event_stream(run)]

    chunks = asyncio.run(collect())
    assert sum("answer" in chunk for chunk in chunks) == 1
    assert chunks[-1].startswith("event: close")


def test_two_connections_each_receive_every_event(tmp_path):
    store = RunStore(tmp_path)
    run = store.create({})
    run.publish({"type": "thought", "data": {}, "t": 0.0})

    async def collect():
        return [chunk async for chunk in event_stream(run)]

    async def both():
        first = asyncio.create_task(collect())
        second = asyncio.create_task(collect())
        await asyncio.sleep(0.05)
        run.publish({"type": "answer", "data": {}, "t": 0.1})
        run.close()
        return await first, await second

    first, second = asyncio.run(both())
    for chunks in (first, second):
        assert sum('"thought"' in chunk for chunk in chunks) == 1
        assert sum('"answer"' in chunk for chunk in chunks) == 1
        assert chunks[-1].startswith("event: close")
