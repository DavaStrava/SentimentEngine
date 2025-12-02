# Async Architecture Enforcement

## Redis Communication

All Redis stream communication MUST use asynchronous APIs:

- Use the official Python `redis` client with asyncio support (redis-py 4.5+)
- Never use blocking Redis calls in the main event loop

## Consumer Implementation

All stream consumers MUST be implemented as asyncio tasks:

```python
import redis.asyncio as redis

async def consume_audio_frames():
    """Example async consumer pattern"""
    redis_client = redis.from_url("redis://localhost")
    while True:
        # Non-blocking stream read
        messages = await redis_client.xread(
            {"audio_stream": "$"}, 
            block=100
        )
        for stream, message_list in messages:
            for message_id, data in message_list:
                await process_frame(data)

# Start as asyncio task
asyncio.create_task(consume_audio_frames())
```

**Note**: Use `redis.asyncio` from redis-py 4.5+ for async operations. The older `aioredis` package is deprecated.

## Task Management

- Each analysis module (acoustic, visual, linguistic) runs as an independent asyncio task
- The Fusion Engine timer runs as a periodic asyncio task
- Stream Input Manager publishes frames asynchronously without blocking

## Critical Rules

1. **Never use blocking I/O** in async functions
2. **Always use `await`** for Redis operations
3. **Use `asyncio.create_task()`** to spawn concurrent analysis modules
4. **Handle task cancellation** gracefully for clean shutdown

## Rationale

This enforces the core of Task 3.1 and Task 11, guaranteeing low-latency, non-blocking I/O critical for real-time performance (Requirement 9.1). The asynchronous architecture ensures that slow modules (e.g., linguistic analysis) don't block fast modules (e.g., acoustic analysis).
