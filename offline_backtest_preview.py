"""Deterministic, offline demonstration of backtest report presentation.

Run with ``python offline_backtest_preview.py``. This module uses synthetic
fixtures only and has no market-data, dashboard, or storage dependencies.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from backtesting import (
    BacktestInputMetadata,
    build_backtest_report,
    compare_backtest_reports,
    run_chronological_evaluation,
)


DEMO_LABEL = "SYNTHETIC DEMONSTRATION DATA - NOT REAL MARKET DATA OR INVESTMENT ADVICE"


def build_synthetic_preview() -> dict[str, object]:
    """Return frictionless and medium-cost report examples from fixed data."""
    dates = pd.bdate_range("2025-01-06", periods=40)
    # Deliberately artificial values: a repeatable fixture, not downloaded
    # prices and not intended to resemble a particular security.
    phase = np.arange(len(dates), dtype=float)
    prices = pd.Series(
        100 + 0.22 * phase + 2.8 * np.sin(phase / 3.2),
        index=dates,
        name="Adj Close",
    )
    benchmark = pd.Series(
        200 + 0.12 * phase + 1.7 * np.sin(phase / 4.1),
        index=dates,
        name="Synthetic benchmark fixture",
    )
    evaluated_at = dates[-1] + pd.Timedelta(hours=16)
    input_metadata = BacktestInputMetadata(
        source="Synthetic generated price fixture",
        latest_observation_at=dates[-1],
        retrieved_at=evaluated_at,
        evaluated_at=evaluated_at,
        freshness_status="fresh synthetic fixture",
        synthetic=True,
    )

    examples = {}
    for key, cost_scenario in (("frictionless", None), ("medium_cost", "medium")):
        evaluation = run_chronological_evaluation(
            prices,
            "sma_crossover",
            development_fraction=0.60,
            validation_fraction=0.20,
            fast_window=3,
            slow_window=7,
            cost_scenario=cost_scenario,
            benchmark=benchmark,
            benchmark_source="Synthetic benchmark fixture generated in this module",
            input_metadata=input_metadata,
        )
        report = build_backtest_report(
            evaluation["backtest"], periods=evaluation["periods"]
        )
        examples[key] = {
            "label": DEMO_LABEL,
            "strategy_description": "Illustrative moving-average crossover on generated prices; not a strategy recommendation.",
            "report": report,
        }
    comparison_reports = {}
    for name, strategy in (
        ("moving_average_crossover", "sma_crossover"),
        ("buy_and_hold", "buy_and_hold"),
    ):
        evaluation = run_chronological_evaluation(
            prices,
            strategy,
            development_fraction=0.60,
            validation_fraction=0.20,
            fast_window=3,
            slow_window=7,
            cost_scenario="medium",
            benchmark=benchmark,
            benchmark_source="Synthetic benchmark fixture generated in this module",
            input_metadata=input_metadata,
        )
        comparison_reports[name] = build_backtest_report(
            evaluation["backtest"], periods=evaluation["periods"]
        )
    comparison = compare_backtest_reports(
        comparison_reports,
        period_start=dates[0],
        period_end=dates[-1],
    )
    return {"label": DEMO_LABEL, "examples": examples, "comparison": comparison}


def main() -> None:
    print(json.dumps(build_synthetic_preview(), indent=2, default=str))


if __name__ == "__main__":
    main()
