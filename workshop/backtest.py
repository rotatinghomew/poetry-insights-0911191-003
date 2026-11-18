"""点-in-time 回测：只用当时排期可见的记录做预测，再与之后实际发生的对比。

纪律
----
- 预测输入截止到 as_of：事件的可见时点（recorded_at 优先，否则 event_time）
  晚于 as_of 的不可见——事后补登的回应不会进入过去的预测。
- 出席/缺席事件永远不用于预测输入，只在评估实际值时作为到场参考。
- 训练/评估周按事件时点所在的周界（同一周的 event_time 共享百位前缀）划分。
- 实际办过的安排可以回测；评估周若满额（删失），实际只给下界，按区间下界相容性
  打分；未办过的方案不在本模块内回测，只能报估计与假设。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .dataio import BookingEvent, Request, Session
from .demand import estimate_demand
from .reconstruct import ATTENDED, NO_SHOW, reconstruct

@dataclass
class SlotBacktest:
    slot: str
    predicted_mean: float | None
    ci_low: float | None
    ci_high: float | None
    actual_observed: int
    actual_censored: bool
    actual_anonymous_views: int
    actual_attendance: int | None
    grade: str
    hit: bool | None          # 真值（或下界）是否与区间相容
    note: str = ""


@dataclass
class BacktestReport:
    as_of: int
    train_weeks: list[str]
    eval_weeks: list[str]
    excluded_late_events: list[str]
    slots: list[SlotBacktest] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "as_of": self.as_of,
            "train_weeks": self.train_weeks,
            "eval_weeks": self.eval_weeks,
            "excluded_late_events": self.excluded_late_events,
            "slots": [
                {
                    "slot": s.slot,
                    "predicted_weekly_demand": s.predicted_mean,
                    "ci90": [s.ci_low, s.ci_high],
                    "actual_observed_demand": s.actual_observed,
                    "actual_censored": s.actual_censored,
                    "actual_anonymous_full_views": s.actual_anonymous_views,
                    "actual_attendance": s.actual_attendance,
                    "grade": s.grade,
                    "interval_consistent": s.hit,
                    "note": s.note,
                }
                for s in self.slots
            ],
        }


def _visible_at(ev: BookingEvent, as_of: int) -> bool:
    if ev.recorded_at is not None:
        return ev.recorded_at <= as_of
    return ev.event_time <= as_of


def run_backtest(
    requests: list[Request],
    events: list[BookingEvent],
    sessions: list[Session],
    as_of: int,
    *,
    eval_weeks: list[str] | None = None,
) -> BacktestReport:
    # 周界：同一周事件的 event_time 共享百位前缀（W1: 100-199, W2: 200-299 ...）
    week_origin = {w: (i + 1) * 100
                   for i, w in enumerate(sorted({s.week for s in sessions}))}

    def week_of_time(t: int) -> str | None:
        idx = t // 100 - 1
        weeks = sorted(week_origin, key=lambda w: week_origin[w])
        return weeks[idx] if 0 <= idx < len(weeks) else None

    all_weeks = sorted({s.week for s in sessions})
    train_weeks = [w for w in all_weeks if (week_origin[w] + 99) <= as_of]
    if eval_weeks is None:
        eval_weeks = [w for w in all_weeks if w not in train_weeks]

    late_ids = [e.event_id for e in events
                if not _visible_at(e, as_of) and e.event not in {ATTENDED, NO_SHOW}]
    visible_events = [e for e in events
                      if _visible_at(e, as_of)
                      and e.event not in {ATTENDED, NO_SHOW}]
    # 申请记录按其归属周是否落在训练窗口内过滤（visible_at 优先）
    week_index = {w: i for i, w in enumerate(all_weeks)}
    train_idx = {week_index[w] for w in train_weeks}
    visible_requests = [
        r for r in requests
        if (r.visible_at is None or r.visible_at <= as_of)
        and week_index.get(r.week, -1) in train_idx]

    recon = reconstruct(visible_requests, visible_events, sessions)
    estimates = estimate_demand(recon)

    # 实际值：用全部数据重建（含事后补登与出席），仅用于打分，不回流入预测
    actual_recon = reconstruct(requests, events, sessions)

    out: list[SlotBacktest] = []
    slots = sorted({s.slot for s in sessions})
    for slot in slots:
        est = estimates.get(slot)
        eval_obs = [actual_recon.obs(w, slot) for w in eval_weeks
                    if (w, slot) in actual_recon.observations]
        if not eval_obs:
            continue
        actual_d = sum(o.observed_demand for o in eval_obs)
        censored = any(o.censored for o in eval_obs)
        anon = sum(o.anonymous_full_views for o in eval_obs)
        attendance = sum(o.attended for o in eval_obs) or None
        if est is None or est.point is None:
            out.append(SlotBacktest(slot, None, None, None, actual_d, censored,
                                    anon, attendance, "C", None,
                                    "训练窗口内无观测，无法给出数据支持的预测"))
            continue
        lo, hi = est.ci_low, est.ci_high
        # 逐周相容性：未删失周要求真值落入区间；删失周只要求区间下界不高于观测下界
        hits: list[bool] = []
        n_cens = 0
        for o in eval_obs:
            if o.closed:
                continue  # 闭馆周无实际值可比
            if o.censored:
                n_cens += 1
                hits.append(lo <= o.observed_demand)
            else:
                hits.append(lo <= o.observed_demand <= hi)
        hit = all(hits) if hits else None
        note = ""
        if n_cens:
            note = (f"{n_cens} 个评估周满额/零名额，实际需求只有下界，"
                    "按区间下界相容性打分")
        out.append(SlotBacktest(slot, round(est.point, 3),
                                round(lo, 3), round(hi, 3),
                                actual_d, censored, anon, attendance,
                                est.grade, hit, note))

    return BacktestReport(
        as_of=as_of, train_weeks=train_weeks, eval_weeks=eval_weeks,
        excluded_late_events=late_ids, slots=out,
    )
