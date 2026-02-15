"""Tests for WebSocket connection manager."""

import asyncio

import pytest

from dashboard.ws_manager import ConnectionManager
from dashboard.event_emitter import DashboardEventEmitter


def test_emitter_subscribe():
    emitter = DashboardEventEmitter()
    q = emitter.subscribe()
    assert q is not None
    assert q.empty()


@pytest.mark.asyncio
async def test_emitter_emit():
    emitter = DashboardEventEmitter()
    q = emitter.subscribe()
    await emitter.emit("test", {"value": 42})
    msg = q.get_nowait()
    assert msg["channel"] == "test"
    assert msg["data"]["value"] == 42
    assert "ts" in msg


def test_emitter_unsubscribe():
    emitter = DashboardEventEmitter()
    q = emitter.subscribe()
    emitter.unsubscribe(q)
    emitter.emit_sync("test", {"x": 1})
    assert q.empty()


def test_emitter_sync():
    emitter = DashboardEventEmitter()
    q = emitter.subscribe()
    emitter.emit_sync("sync_test", {"hello": "world"})
    msg = q.get_nowait()
    assert msg["channel"] == "sync_test"


def test_connection_manager_init():
    mgr = ConnectionManager()
    assert mgr.active_count == 0
