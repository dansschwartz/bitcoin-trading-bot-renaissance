"""Tests for agents.event_bus.EventBus."""

import threading
import time
import pytest

# Adjust path so tests can find the agents package
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.event_bus import EventBus


class TestEventBus:
    """EventBus subscribe/emit, wildcards, error handling."""

    def test_basic_subscribe_and_emit(self):
        bus = EventBus()
        received = []
        bus.subscribe("test.event", lambda ch, data: received.append((ch, data)))
        bus.emit("test.event", {"value": 42})
        assert len(received) == 1
        assert received[0] == ("test.event", {"value": 42})

    def test_wildcard_subscribe(self):
        bus = EventBus()
        received = []
        bus.subscribe("risk.*", lambda ch, data: received.append(ch))
        bus.emit("risk.alert", {"level": "high"})
        bus.emit("risk.circuit_breaker", {"reason": "test"})
        bus.emit("signal.update", {"value": 1})  # should NOT match
        assert received == ["risk.alert", "risk.circuit_breaker"]

    def test_star_matches_all(self):
        bus = EventBus()
        received = []
        bus.subscribe("*", lambda ch, data: received.append(ch))
        bus.emit("foo", {})
        bus.emit("bar.baz", {})
        assert len(received) == 2

    def test_unsubscribe(self):
        bus = EventBus()
        received = []
        handler = lambda ch, data: received.append(ch)
        bus.subscribe("test", handler)
        bus.emit("test", {})
        assert len(received) == 1
        bus.unsubscribe("test", handler)
        bus.emit("test", {})
        assert len(received) == 1  # no new event

    def test_handler_error_does_not_break_others(self):
        bus = EventBus()
        results = []

        def bad_handler(ch, data):
            raise ValueError("boom")

        def good_handler(ch, data):
            results.append("ok")

        bus.subscribe("test", bad_handler, subscriber="bad")
        bus.subscribe("test", good_handler, subscriber="good")
        bus.emit("test", {})
        assert results == ["ok"]

    def test_history(self):
        bus = EventBus(max_history=5)
        for i in range(10):
            bus.emit("tick", {"i": i})
        history = bus.get_history("tick", limit=100)
        # Ring buffer keeps last 5
        assert len(history) == 5
        # Newest first
        assert history[0]["data"]["i"] == 9

    def test_history_with_pattern(self):
        bus = EventBus()
        bus.emit("risk.alert", {"a": 1})
        bus.emit("signal.update", {"b": 2})
        bus.emit("risk.circuit", {"c": 3})
        risk_events = bus.get_history("risk.*")
        assert len(risk_events) == 2

    def test_get_cached(self):
        bus = EventBus()
        assert bus.get_cached("test") is None
        bus.emit("test", {"value": 123})
        assert bus.get_cached("test") == {"value": 123}
        bus.emit("test", {"value": 456})
        assert bus.get_cached("test") == {"value": 456}

    def test_thread_safety(self):
        bus = EventBus()
        results = []
        bus.subscribe("count", lambda ch, data: results.append(data["n"]))

        def emit_range(start, end):
            for i in range(start, end):
                bus.emit("count", {"n": i})

        threads = [threading.Thread(target=emit_range, args=(i * 100, (i + 1) * 100)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 500

    def test_clear_history(self):
        bus = EventBus()
        bus.emit("test", {"a": 1})
        assert len(bus.get_history()) == 1
        bus.clear_history()
        assert len(bus.get_history()) == 0
        assert bus.get_cached("test") is None

    def test_multiple_handlers_same_channel(self):
        bus = EventBus()
        r1, r2 = [], []
        bus.subscribe("ch", lambda c, d: r1.append(1))
        bus.subscribe("ch", lambda c, d: r2.append(2))
        bus.emit("ch", {})
        assert r1 == [1]
        assert r2 == [2]
