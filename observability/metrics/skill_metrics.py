"""
Skill-level success rates and latency tracking.
In-process counters — export to Prometheus, Datadog, or OTEL Metrics as needed.

Usage:
    from observability.metrics.skill_metrics import record, summary
    record("fingerprint", success=True, duration_ms=4.2, batch_size=10000)
    print(summary())
"""
from __future__ import annotations
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class SkillStats:
    calls: int = 0
    successes: int = 0
    failures: int = 0
    total_ms: float = 0.0
    total_items: int = 0
    _p99_reservoir: list = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.successes / self.calls if self.calls else 0.0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.calls if self.calls else 0.0

    @property
    def throughput(self) -> float:
        """items / second"""
        total_s = self.total_ms / 1000
        return self.total_items / total_s if total_s > 0 else 0.0


_stats: Dict[str, SkillStats] = defaultdict(SkillStats)


def record(
    skill: str,
    success: bool = True,
    duration_ms: float = 0.0,
    batch_size: int = 1,
) -> None:
    s = _stats[skill]
    s.calls += 1
    s.total_ms += duration_ms
    s.total_items += batch_size
    if success:
        s.successes += 1
    else:
        s.failures += 1


def summary() -> str:
    lines = [f"{'Skill':<30} {'Calls':>6} {'Success%':>9} {'AvgMs':>8} {'Throughput':>14}"]
    lines.append("-" * 75)
    for skill, s in sorted(_stats.items()):
        lines.append(
            f"{skill:<30} {s.calls:>6} {s.success_rate*100:>8.1f}% "
            f"{s.avg_ms:>7.1f}ms {s.throughput:>12,.0f}/s"
        )
    return "\n".join(lines)


class timed:
    """Context manager that auto-records a skill call."""
    def __init__(self, skill: str, batch_size: int = 1):
        self.skill = skill
        self.batch_size = batch_size
        self.success = True

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def fail(self):
        self.success = False

    def __exit__(self, exc_type, *_):
        if exc_type is not None:
            self.success = False
        duration_ms = (time.perf_counter() - self._t0) * 1000
        record(self.skill, success=self.success, duration_ms=duration_ms, batch_size=self.batch_size)
