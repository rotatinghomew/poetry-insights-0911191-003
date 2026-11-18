"""候选增场方案的联合评估。

为什么不能把每场分别预测的预约数相加：
同一位申请人只应被服务一次，新增场会从旧场分流，旧场也会接住新增场满额后的人。
本模块在每一轮蒙特卡洛里生成整盘申请人，再按各自偏好序做联合分配（随机顺序
serial dictatorship），同一批随机数同时跑"只有旧场"的基线，两盘相减才是
方案的净新增与挤出。

硬约束（违反即方案不可执行）：
- 同一讲师同一时段不得带两场；同一教室同一时段不得排两场；
- 最低开课人数不得大于名额；session_id 唯一；
- 场次时段必须落在需求模型已知的可接受时段集合内，或在 assumptions 中显式声明。
最低开课人数不足导致整场取消、已分配者重新找座，是"挤出"的主要来源。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .dataio import Plan, PlanSession
from .demand import GRADE_C, SlotDemandEstimate
from .prefs import PreferenceModel


@dataclass
class Conflict:
    kind: str
    detail: str
    sessions: list[str] = field(default_factory=list)


@dataclass
class SessionStats:
    session_id: str
    slot: str
    is_added: bool
    capacity: int
    minimum: int
    expected_enrolled: float
    ci_enrolled: tuple[float, float]
    prob_meets_minimum: float
    expected_attendance: float | None
    expected_turnaway: float
    cancel_rate: float


@dataclass
class PlanEvaluation:
    plan_name: str
    feasible: bool
    conflicts: list[Conflict]
    expected_served: float
    ci_served: tuple[float, float]
    expected_unserved: float
    expected_attendance: float | None
    newly_served_vs_baseline: float
    displaced_vs_baseline: float
    per_session: list[SessionStats]
    substitution_matrix: dict[str, dict[str, float]]  # 首选时段 -> 入座时段 -> 期望人数
    diversion_old_to_new: dict[str, dict[str, float]]
    evidence_gaps: list[str]
    assumptions: dict[str, float]
    n_runs: int

    def as_dict(self) -> dict:
        return {
            "plan": self.plan_name,
            "feasible": self.feasible,
            "conflicts": [c.__dict__ for c in self.conflicts],
            "expected_served": round(self.expected_served, 3),
            "ci90_served": [round(self.ci_served[0], 3), round(self.ci_served[1], 3)],
            "expected_unserved": round(self.expected_unserved, 3),
            "expected_attendance": (
                None if self.expected_attendance is None
                else round(self.expected_attendance, 3)),
            "newly_served_vs_baseline": round(self.newly_served_vs_baseline, 3),
            "displaced_vs_baseline": round(self.displaced_vs_baseline, 3),
            "per_session": [
                {
                    "session_id": s.session_id,
                    "slot": s.slot,
                    "is_added": s.is_added,
                    "capacity": s.capacity,
                    "minimum": s.minimum,
                    "expected_enrolled": round(s.expected_enrolled, 3),
                    "ci90_enrolled": [round(s.ci_enrolled[0], 3),
                                      round(s.ci_enrolled[1], 3)],
                    "prob_meets_minimum": round(s.prob_meets_minimum, 3),
                    "expected_attendance": (
                        None if s.expected_attendance is None
                        else round(s.expected_attendance, 3)),
                    "expected_turnaway": round(s.expected_turnaway, 3),
                    "cancel_rate": round(s.cancel_rate, 3),
                }
                for s in self.per_session
            ],
            "substitution_matrix": {
                k: {kk: round(vv, 3) for kk, vv in row.items()}
                for k, row in self.substitution_matrix.items()
            },
            "diversion_old_to_new": {
                k: {kk: round(vv, 3) for kk, vv in row.items()}
                for k, row in self.diversion_old_to_new.items()
            },
            "evidence_gaps": self.evidence_gaps,
            "assumptions": {k: round(v, 3) for k, v in self.assumptions.items()},
            "monte_carlo_runs": self.n_runs,
        }


# ---- 硬约束 -----------------------------------------------------------------

def check_constraints(plan: Plan, known_slots: set[str]) -> list[Conflict]:
    conflicts: list[Conflict] = []
    sessions = plan.all_sessions
    ids = [s.session_id for s in sessions]
    if len(ids) != len(set(ids)):
        dup = sorted({x for x in ids if ids.count(x) > 1})
        conflicts.append(Conflict("duplicate_session_id",
                                  f"场次编号重复: {dup}", dup))
    by_slot: dict[str, list[PlanSession]] = {}
    for s in sessions:
        by_slot.setdefault(s.slot, []).append(s)
        if s.minimum > s.capacity:
            conflicts.append(Conflict(
                "minimum_above_capacity",
                f"{s.session_id} 最低开课人数 {s.minimum} 大于名额 {s.capacity}",
                [s.session_id]))
        if s.capacity <= 0:
            conflicts.append(Conflict(
                "nonpositive_capacity",
                f"{s.session_id} 名额为 {s.capacity}，无法开课", [s.session_id]))
        if s.slot not in known_slots and not s.acceptable_slots:
            conflicts.append(Conflict(
                "unknown_slot",
                f"{s.session_id} 的时段 {s.slot} 不在历史可接受时段 {sorted(known_slots)} 内，"
                "且未声明 acceptable_slots 与需求假设", [s.session_id]))
    for slot, group in by_slot.items():
        teachers = [s.teacher for s in group]
        rooms = [s.room for s in group]
        for i, s1 in enumerate(group):
            for s2 in group[i + 1:]:
                if s1.teacher == s2.teacher:
                    conflicts.append(Conflict(
                        "teacher_conflict",
                        f"{slot} 讲师 {s1.teacher} 同时带 {s1.session_id} 与 {s2.session_id}",
                        [s1.session_id, s2.session_id]))
                if s1.room == s2.room:
                    conflicts.append(Conflict(
                        "room_conflict",
                        f"{slot} 教室 {s1.room} 同时排 {s1.session_id} 与 {s2.session_id}",
                        [s1.session_id, s2.session_id]))
    return conflicts


# ---- 单轮分配 ---------------------------------------------------------------

@dataclass
class _Applicant:
    aid: int
    rank1_slot: str
    preference: list[str]


def _resolve_lambdas(
    plan: Plan,
    estimates: dict[str, SlotDemandEstimate],
    assumptions: dict[str, float],
) -> tuple[dict[str, float], list[str]]:
    slots = {s.slot for s in plan.all_sessions}
    lambdas: dict[str, float] = {}
    gaps: list[str] = []
    for slot in sorted(slots):
        est = estimates.get(slot)
        if slot in assumptions:
            lambdas[slot] = float(assumptions[slot])
            continue
        if est is not None and est.point is not None and est.grade != GRADE_C:
            lambdas[slot] = est.point
        elif est is not None and est.point is not None:
            # C 级但有插补点估计：默认不用于排期，除非显式 assumption
            gaps.append(f"{slot} 证据等级 C（{'；'.join(est.grade_reasons)}），"
                        "需在 assumptions 中显式给出需求假设并先试开")
        else:
            gaps.append(f"{slot} 没有任何历史需求估计，需在 assumptions 中显式声明")
    return lambdas, gaps


def _draw_applicants(lambdas: dict[str, float],
                     estimates: dict[str, SlotDemandEstimate],
                     assumptions: dict[str, float],
                     prefs: PreferenceModel,
                     rng: random.Random) -> list[_Applicant]:
    applicants: list[_Applicant] = []
    aid = 0
    for slot, lam in lambdas.items():
        # 层级不确定性：有后验参数且非人工假设时，先抽本轮 λ
        est = estimates.get(slot)
        if slot not in assumptions and est is not None and est.posterior_alpha:
            lam_run = _gamma_draw(est.posterior_alpha, est.posterior_beta, rng)
        else:
            lam_run = lam
        n = _poisson_draw(lam_run, rng)
        for _ in range(n):
            order = prefs.sample_preference_order(slot, rng)
            applicants.append(_Applicant(aid, slot, order))
            aid += 1
    return applicants


def _gamma_draw(shape: float, rate: float, rng: random.Random) -> float:
    return rng.gammavariate(shape, 1.0 / rate)


def _poisson_draw(lam: float, rng: random.Random) -> int:
    # Knuth，对小 λ 足够；排期场景 λ 通常不大
    if lam <= 0:
        return 0
    l = math.exp(-lam)
    k, p = 0, 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= l:
            return k - 1


def _allocate(
    sessions: list[PlanSession],
    applicants: list[_Applicant],
    rng: random.Random,
    existing_ids: set[str] | None = None,
) -> tuple[dict[str, int], dict[int, str], set[str], dict[str, int]]:
    """返回 每场入座人数、申请人入座场次、被取消场次、每场被拒人数。"""
    existing_ids = existing_ids or set()
    remaining = {s.session_id: s.capacity for s in sessions}
    # 时段 -> 可承接该时段需求的场次；
    # 同时段多场时旧场（existing）优先承接，避免新增场无谓挤掉旧场入座者，
    # 同优先级再按 session_id 保证确定性。
    acceptors: dict[str, list[PlanSession]] = {}
    for s in sessions:
        acceptable = s.acceptable_slots or [s.slot]
        for slot in acceptable:
            acceptors.setdefault(slot, []).append(s)
    for slot in acceptors:
        acceptors[slot].sort(key=lambda s: (
            0 if s.session_id in existing_ids else 1, s.session_id))

    seated: dict[int, str] = {}
    order = list(applicants)
    rng.shuffle(order)
    for ap in order:
        for wanted in ap.preference:
            for s in acceptors.get(wanted, []):
                if remaining[s.session_id] <= 0:
                    continue
                remaining[s.session_id] -= 1
                seated[ap.aid] = s.session_id
                break
            if ap.aid in seated:
                break

    # 最低开课人数：不足则取消并重新安置
    enrolled = {sid: 0 for sid in remaining}
    for sid in seated.values():
        enrolled[sid] += 1
    cancelled: set[str] = set()
    for s in sessions:
        if enrolled[s.session_id] < s.minimum:
            cancelled.add(s.session_id)
    turnaway: dict[str, int] = {s.session_id: 0 for s in sessions}
    if cancelled:
        evicted = [(ap, seated[ap.aid])
                   for ap in applicants if seated.get(ap.aid) in cancelled]
        for ap, _old_sid in evicted:
            del seated[ap.aid]
        rng.shuffle(evicted)
        for ap, old_sid in evicted:
            placed = False
            for wanted in ap.preference:
                for s in acceptors.get(wanted, []):
                    if s.session_id in cancelled:
                        continue
                    if remaining[s.session_id] > 0:
                        remaining[s.session_id] -= 1
                        seated[ap.aid] = s.session_id
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                turnaway[old_sid] = turnaway.get(old_sid, 0) + 1
    enrolled = {sid: 0 for sid in remaining}
    for sid in seated.values():
        enrolled[sid] += 1
    return enrolled, seated, cancelled, turnaway


# ---- 主评估 -----------------------------------------------------------------

def evaluate_plan(
    plan: Plan,
    estimates: dict[str, SlotDemandEstimate],
    prefs: PreferenceModel,
    *,
    n_runs: int = 1000,
    seed: int = 20261118,
    assumptions: dict[str, float] | None = None,
    show_rates: dict[str, float] | None = None,
    known_slots: set[str] | None = None,
) -> PlanEvaluation:
    assumptions = dict(assumptions or {})
    show_rates = show_rates or {}
    known_slots = known_slots if known_slots is not None else set(estimates)
    conflicts = check_constraints(plan, known_slots)

    lambdas, gaps = _resolve_lambdas(plan, estimates, assumptions)
    if conflicts:
        return PlanEvaluation(
            plan_name=plan.name, feasible=False, conflicts=conflicts,
            expected_served=0.0, ci_served=(0.0, 0.0), expected_unserved=0.0,
            expected_attendance=None, newly_served_vs_baseline=0.0,
            displaced_vs_baseline=0.0, per_session=[],
            substitution_matrix={}, diversion_old_to_new={},
            evidence_gaps=gaps, assumptions=assumptions, n_runs=0)

    rng = random.Random(seed)
    sessions = plan.all_sessions
    added_ids = {s.session_id for s in plan.added}
    existing_ids = {s.session_id for s in plan.existing}
    baseline = Plan(name=f"{plan.name}#baseline", existing=plan.existing, added=[])

    # 累计器
    served_runs: list[float] = []
    att_runs: list[float] = []
    enroll_acc = {s.session_id: 0.0 for s in sessions}
    enroll_runs = {s.session_id: [] for s in sessions}
    min_hit = {s.session_id: 0 for s in sessions}
    cancel_acc = {s.session_id: 0 for s in sessions}
    turn_acc = {s.session_id: 0.0 for s in sessions}
    sub_acc: dict[str, dict[str, float]] = {}
    divert_acc: dict[str, dict[str, float]] = {}
    newly_acc, displaced_acc = 0.0, 0.0

    sid_by_slot = {s.session_id: s.slot for s in sessions}

    for _ in range(n_runs):
        applicants = _draw_applicants(lambdas, estimates, assumptions, prefs, rng)
        enroll, seated, cancelled, turnaway = _allocate(
            sessions, applicants, rng, existing_ids)
        _base_enroll, base_seated, _bc, _bt = _allocate(
            baseline.all_sessions, applicants, rng, existing_ids)

        served = len(seated)
        served_runs.append(served)
        att = sum(enroll[sid] * show_rates.get(sid_by_slot[sid], 1.0)
                  for sid in enroll)
        att_runs.append(att)
        newly_acc += sum(1 for aid, sid in seated.items()
                         if aid not in base_seated)
        displaced_acc += sum(1 for aid in base_seated if aid not in seated)

        for s in sessions:
            sid = s.session_id
            enroll_acc[sid] += enroll[sid]
            enroll_runs[sid].append(enroll[sid])
            if sid not in cancelled and enroll[sid] >= s.minimum:
                min_hit[sid] += 1
            if sid in cancelled:
                cancel_acc[sid] += 1
            turn_acc[sid] += turnaway.get(sid, 0)

        for ap in applicants:
            sid = seated.get(ap.aid)
            if sid is None:
                continue
            got_slot = sid_by_slot[sid]
            sub_acc.setdefault(ap.rank1_slot, {})
            sub_acc[ap.rank1_slot][got_slot] = \
                sub_acc[ap.rank1_slot].get(got_slot, 0.0) + 1.0
            if sid in added_ids and ap.rank1_slot != got_slot:
                divert_acc.setdefault(ap.rank1_slot, {})
                divert_acc[ap.rank1_slot][got_slot] = \
                    divert_acc[ap.rank1_slot].get(got_slot, 0.0) + 1.0

    def _quantile(data: list[float], q: float) -> float:
        if not data:
            return 0.0
        xs = sorted(data)
        idx = min(len(xs) - 1, max(0, round(q * (len(xs) - 1))))
        return xs[idx]

    per_session: list[SessionStats] = []
    for s in sessions:
        sid = s.session_id
        enr = enroll_acc[sid] / n_runs
        per_session.append(SessionStats(
            session_id=sid, slot=s.slot, is_added=sid in added_ids,
            capacity=s.capacity, minimum=s.minimum,
            expected_enrolled=enr,
            ci_enrolled=(_quantile(enroll_runs[sid], 0.05),
                         _quantile(enroll_runs[sid], 0.95)),
            prob_meets_minimum=min_hit[sid] / n_runs,
            expected_attendance=enr * show_rates.get(s.slot, 1.0)
            if show_rates else None,
            expected_turnaway=turn_acc[sid] / n_runs,
            cancel_rate=cancel_acc[sid] / n_runs,
        ))

    return PlanEvaluation(
        plan_name=plan.name, feasible=True, conflicts=[],
        expected_served=sum(served_runs) / n_runs,
        ci_served=(_quantile(served_runs, 0.05),
                   _quantile(served_runs, 0.95)),
        expected_unserved=max(0.0, sum(lambdas.values())
                              - sum(served_runs) / n_runs),
        expected_attendance=sum(att_runs) / n_runs if show_rates else None,
        newly_served_vs_baseline=newly_acc / n_runs,
        displaced_vs_baseline=displaced_acc / n_runs,
        per_session=per_session,
        substitution_matrix={k: {kk: v / n_runs for kk, v in row.items()}
                             for k, row in sub_acc.items()},
        diversion_old_to_new={k: {kk: v / n_runs for kk, v in row.items()}
                              for k, row in divert_acc.items()},
        evidence_gaps=gaps, assumptions=assumptions, n_runs=n_runs,
    )


