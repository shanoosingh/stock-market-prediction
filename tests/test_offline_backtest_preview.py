import builtins
import socket
import sqlite3

import pytest

from offline_backtest_preview import DEMO_LABEL, build_synthetic_preview


def test_preview_is_deterministic_and_contains_both_cost_modes():
    first = build_synthetic_preview()
    second = build_synthetic_preview()

    assert first == second
    assert first["label"] == DEMO_LABEL
    assert set(first["examples"]) == {"frictionless", "medium_cost"}
    frictionless = first["examples"]["frictionless"]
    charged = first["examples"]["medium_cost"]
    assert frictionless["label"] == charged["label"] == DEMO_LABEL
    assert frictionless["report"]["cost_description"] == "ZERO COST / FRICTIONLESS"
    assert charged["report"]["cost_description"].startswith(
        "Indian delivery scenario: medium;"
    )
    for example in (frictionless, charged):
        report = example["report"]
        assert "gross_cumulative_return" in report
        assert "net_cumulative_return" in report
        assert report["benchmark"]["source"] == (
            "Synthetic benchmark fixture generated in this module"
        )
        assert report["benchmark"]["date_policy"] == "Exact common price dates; no filling"
        assert report["benchmark"]["matched_return_intervals"] == 39
        assert set(report["periods"]) == {
            "development",
            "validation",
            "final_evaluation",
        }
        assert "not a forecast" in " ".join(report["limitations"])
        assert "not a strategy recommendation" in example["strategy_description"]
    comparison = first["comparison"]
    assert comparison["comparable"] is True
    assert comparison["warnings"] == []
    assert [row["name"] for row in comparison["strategies"]] == [
        "moving_average_crossover",
        "buy_and_hold",
    ]
    assert comparison["ranking"] is None
    assert "SYNTHETIC DEMONSTRATION DATA" in first["label"]


def test_preview_does_not_import_providers_open_network_or_access_sqlite(monkeypatch):
    original_import = builtins.__import__
    blocked = (
        "market_data_service",
        "market_data_storage",
        "yfinance",
        "openchart",
        "requests",
        "urllib",
    )

    def guarded_import(name, *args, **kwargs):
        if any(name == module or name.startswith(module + ".") for module in blocked):
            pytest.fail(f"Offline preview attempted to import {name}.")
        return original_import(name, *args, **kwargs)

    def fail_if_called(*args, **kwargs):
        pytest.fail("Offline preview attempted network or SQLite access.")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(socket.socket, "connect", fail_if_called)
    monkeypatch.setattr(socket, "create_connection", fail_if_called)
    monkeypatch.setattr(sqlite3, "connect", fail_if_called)

    result = build_synthetic_preview()

    assert result["label"] == DEMO_LABEL
