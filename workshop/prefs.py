"""偏好与替代模型：申请人接受哪些时段、按什么顺序退而求其次。

证据来源（全部是同一申请人在同一周内的行为，转场不算新增需求）：
- requests 的偏好顺序；
- 候补后接受另一场（accepted_diversion）；
- 撤销后回流到另一场（reflow）。
样本不足的 from_slot 回退到全局热度序，并在报告中标注为回退假设。
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field

from .dataio import Request
from .reconstruct import Reconstruction


@dataclass
class PreferenceModel:
    rank1_counts: dict[str, int]
    transitions: dict[str, dict[str, int]]  # from_slot -> to_slot -> 证据条数
    fallback_order: list[str]
    weak_slots: set[str] = field(default_factory=set)

    def sample_preference_order(self, rank1_slot: str,
                                rng: random.Random) -> list[str]:
        """给定首选时段，抽一个该申请人的完整偏好序（含首选）。"""
        order = [rank1_slot]
        row = self.transitions.get(rank1_slot, {})
        if row:
            slots = list(row)
            weights = [row[s] for s in slots]
            while slots:
                pick = _weighted_without_replacement(slots, weights, rng)
                if pick not in order:
                    order.append(pick)
                idx = slots.index(pick)
                slots.pop(idx)
                weights.pop(idx)
        # 回退：全局热度序补全
        for slot in self.fallback_order:
            if slot not in order:
                order.append(slot)
        return order


def _weighted_without_replacement(slots: list[str], weights: list[float],
                                  rng: random.Random) -> str:
    total = sum(weights)
    r = rng.random() * total
    upto = 0.0
    for slot, w in zip(slots, weights):
        upto += w
        if upto >= r:
            return slot
    return slots[-1]


def build_preference_model(
    requests: list[Request],
    recon: Reconstruction | None = None,
) -> PreferenceModel:
    rank1_counts: dict[str, int] = defaultdict(int)
    transitions: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    by_person: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
    for r in requests:
        by_person[(r.applicant, r.week)].append((r.rank, r.slot))

    for (_p, _w), entries in by_person.items():
        entries.sort()
        first = entries[0][1]
        rank1_counts[first] += 1
        for _rank, later_slot in entries[1:]:
            if later_slot != first:
                transitions[first][later_slot] += 1

    if recon is not None:
        for f in recon.flows:
            if f.from_slot != f.to_slot:
                transitions[f.from_slot][f.to_slot] += 1

    popularity = sorted(rank1_counts, key=lambda s: (-rank1_counts[s], s))
    all_slots = set(popularity) | set(transitions)
    fallback = popularity + sorted(all_slots - set(popularity))

    weak = {s for s, row in transitions.items() if sum(row.values()) < 2}
    return PreferenceModel(
        rank1_counts=dict(rank1_counts),
        transitions={s: dict(row) for s, row in transitions.items()},
        fallback_order=fallback,
        weak_slots=weak,
    )
