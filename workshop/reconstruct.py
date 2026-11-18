"""把申请偏好与预约流水重建成逐周逐时段的需求观测。

分类口径（对应排期组的五类信号）：

1. 有凭据的首次需求：requests 中 rank==1 的去重申请人，是独立需求的计数基础。
2. 改报替代场：同一申请人 rank>1 的偏好，或候补后接受另一场；只算替代证据，
   不计入任何时段的新增独立需求（进入偏好替代模型，见 prefs.py）。
3. 退订回流：申请人撤销 A 后在同一周确认/接受 B，记录 A→B 回流，不算新增。
4. 重复通知：同一申请人收到多次候补邀请；只计一次人，重复次数单列。
5. 没有观测的需求：满额后的匿名售罄浏览（无 request 身份）只作"存在未知需求"
   的证据；闭馆时段完全没有需求观测；名额设零是另一种删失观测，二者严格区分。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .dataio import BookingEvent, Request, Session

# 事件类型常量
CONFIRM = "确认"
INVITE = "候补邀请"
ACCEPT = "接受"
DECLINE = "拒绝"
CANCEL = "撤销"
FULL_VIEW = "售罄浏览"
CLOSURE = "闭馆"
CAPACITY_CHANGE = "名额变动"
ATTENDED = "出席"
NO_SHOW = "缺席"

KNOWN_EVENTS = {
    CONFIRM, INVITE, ACCEPT, DECLINE, CANCEL,
    FULL_VIEW, CLOSURE, CAPACITY_CHANGE, ATTENDED, NO_SHOW,
}

FlowKind = Literal["preference_diversion", "accepted_diversion", "reflow"]


@dataclass(frozen=True)
class Flow:
    """同一申请人在同一周内跨时段的流动。"""

    applicant: str
    week: str
    from_slot: str
    to_slot: str
    kind: FlowKind
    event_time: int | None


@dataclass
class SlotWeekObs:
    week: str
    slot: str
    # 凭据需求
    rank1_applicants: set[str] = field(default_factory=set)
    credentialed_applicants: set[str] = field(default_factory=set)
    diverted_in: set[str] = field(default_factory=set)       # rank>1 才出现的人
    reflow_in: set[str] = field(default_factory=set)
    # 满额与匿名证据
    sold_out: bool = False
    anonymous_full_views: int = 0
    identified_full_views: int = 0
    invitations: int = 0
    duplicate_invites: int = 0
    # 供给状态
    closed: bool = False
    zero_capacity: bool = False
    capacity_known: bool = False
    min_capacity: int | None = None      # 该周见过的最低非空名额
    sellout_capacity: int | None = None  # 首次满额时生效名额（删失阈值 q）
    no_event_records: bool = False
    capacity_unknown: bool = False
    # 结果性数据（预测输入不使用，仅回测实际值用）
    confirmed: set[str] = field(default_factory=set)
    final_waitlist: set[str] = field(default_factory=set)
    attended: int = 0
    no_show: int = 0
    anomalies: list[str] = field(default_factory=list)

    @property
    def censored(self) -> bool:
        """该 slot-week 的到达过程是否在某处被截断。"""
        if self.closed:
            return False  # 闭馆是"无观测"，不是删失
        return self.sold_out or self.zero_capacity

    @property
    def observed_demand(self) -> int:
        """有身份的观测需求人数（rank1，独立需求口径）。"""
        return len(self.rank1_applicants)


@dataclass
class Reconstruction:
    observations: dict[tuple[str, str], SlotWeekObs]
    flows: list[Flow]
    weeks: list[str]
    unassigned_anonymous_events: list[str]
    unknown_event_types: list[str]

    def obs(self, week: str, slot: str) -> SlotWeekObs:
        return self.observations[(week, slot)]

    def slot_weeks(self, slot: str) -> list[SlotWeekObs]:
        return [o for o in self.observations.values() if o.slot == slot]


def _event_week(
    event: BookingEvent,
    request_week: dict[str, str],
) -> str | None:
    if event.week:
        return event.week  # 匿名供给事件（名额变动/闭馆）的显式归属
    if event.request_id and event.request_id in request_week:
        return request_week[event.request_id]
    return None  # 匿名且无 week 列：无法归周


def reconstruct(
    requests: list[Request],
    events: list[BookingEvent],
    sessions: list[Session],
) -> Reconstruction:
    request_by_id = {r.request_id: r for r in requests}
    request_week = {r.request_id: r.week for r in requests}

    # 申请人 -> 周 -> [(rank, slot, request_id)]
    prefs: dict[str, dict[str, list[tuple[int, str, str]]]] = {}
    for r in requests:
        prefs.setdefault(r.applicant, {}).setdefault(r.week, []).append(
            (r.rank, r.slot, r.request_id)
        )

    weeks = sorted({r.week for r in requests} | {s.week for s in sessions})
    known_slots = sorted({r.slot for r in requests} | {s.slot for s in sessions}
                         | {e.slot for e in events})

    obs: dict[tuple[str, str], SlotWeekObs] = {}
    for w in weeks:
        for slot in known_slots:
            obs[(w, slot)] = SlotWeekObs(week=w, slot=slot)

    # sessions 表给出闭馆与"排过但无流水"的信息
    session_keys = {(s.week, s.slot) for s in sessions}
    for s in sessions:
        ob = obs[(s.week, s.slot)]
        if s.closed:
            ob.closed = True

    # 有凭据需求（rank1 独立需求；rank>1 仅替代证据）
    for r in requests:
        ob = obs[(r.week, r.slot)]
        ob.credentialed_applicants.add(r.applicant)
        if r.rank == 1:
            ob.rank1_applicants.add(r.applicant)
        else:
            ob.diverted_in.add(r.applicant)

    flows: list[Flow] = []

    # 偏好层面的替代证据
    for applicant, by_week in prefs.items():
        for week, entries in by_week.items():
            entries.sort()
            rank1 = entries[0]
            for rank, slot, _rid in entries[1:]:
                if rank1[1] != slot:
                    flows.append(Flow(applicant, week, rank1[1], slot,
                                      "preference_diversion", None))

    # 事件驱动状态机：按 (周, 时段) 维护座位与候补
    seats: dict[tuple[str, str], int] = {}
    waitlist: dict[tuple[str, str], list[str]] = {}
    invited: dict[tuple[str, str], set[str]] = {}
    unassigned: list[str] = []
    unknown_types: list[str] = []
    # 申请人周内时间线，用于回流判断
    person_events: dict[tuple[str, str], list[tuple[int, str, str]]] = {}

    for ev in events:
        if ev.event not in KNOWN_EVENTS:
            unknown_types.append(ev.event_id)
            continue
        week = _event_week(ev, request_week)
        if week is None:
            # 匿名事件且无法归周：绝不凭空塞进某个 slot-week
            unassigned.append(ev.event_id)
            continue
        # 事件归槽：带 request 身份的事件按该 request 的偏好时段归属，
        # 与流水里记录的 slot（可能是首选场而非实际发生场）解耦；
        # 匿名事件只能使用流水自身的 slot。
        linked = request_by_id.get(ev.request_id)
        person_slot = linked.slot if linked else ev.slot
        person_key = (week, person_slot)
        supply_key = (week, ev.slot)
        ob = obs[person_key]
        applicant = linked.applicant if linked else ""

        # 名额变动是时段供给属性，一律按流水记录的实际 slot 记账
        # （零名额周的候补邀请常挂在首选场的 request 上，不能据此改首选场名额）；
        # 人员身份则按 request 的偏好时段归属。
        supply_ob = obs[supply_key]
        if ev.event == CLOSURE:
            supply_ob.closed = True
            continue
        if ev.capacity is not None:
            supply_ob.capacity_known = True
            # 满额后人员事件携带的 0 是"剩余名额"，不是供给名额。
            # 正名额事件锚定真实供给；仅在从未出现过正供给名额时，
            # 非人员事件的 0 才算"名额设零"（与闭馆区分）。
            is_person_event = ev.event in {
                INVITE, ACCEPT, DECLINE, FULL_VIEW, ATTENDED, NO_SHOW,
                CONFIRM, CANCEL}
            if ev.capacity > 0:
                supply_ob.min_capacity = (
                    ev.capacity if supply_ob.min_capacity is None
                    else min(supply_ob.min_capacity, ev.capacity))
            elif not is_person_event and supply_ob.min_capacity is None:
                supply_ob.zero_capacity = True
                supply_ob.min_capacity = 0
            # 只有纯供给事件（名额变动）才重置座位计数器；
            # 人员事件携带的 capacity 是当时快照，不能覆盖运行中的余位。
            if ev.event == CAPACITY_CHANGE:
                seats[supply_key] = ev.capacity
            elif supply_key not in seats and ev.capacity > 0:
                seats[supply_key] = ev.capacity  # 没有独立名额变动事件时的兜底
        if ev.event == CAPACITY_CHANGE:
            continue  # 纯供给事件，无人员语义
        if ob.closed or supply_ob.closed:
            # 闭馆之后的事件不构成需求观测
            continue
        if ev.event == ATTENDED:
            if applicant:
                supply_ob.attended += 1
            continue
        if ev.event == NO_SHOW:
            if applicant:
                supply_ob.no_show += 1
            continue

        if applicant:
            person_events.setdefault((applicant, week), []).append(
                (ev.event_time, ev.event, ev.slot))

        if ev.event == FULL_VIEW:
            if seats.get(supply_key, 1) == 0:
                supply_ob.sold_out = True
                if supply_ob.sellout_capacity is None:
                    supply_ob.sellout_capacity = 0
                if applicant:
                    supply_ob.identified_full_views += 1
                else:
                    supply_ob.anonymous_full_views += 1
            continue

        if not applicant:
            # 其余事件没有身份则无法入账，记入异常
            ob.anomalies.append(f"{ev.event_id}: 无身份的{ev.event}")
            continue

        if ev.event == CONFIRM:
            if seats.get(supply_key, 0) > 0:
                if applicant not in supply_ob.confirmed:
                    supply_ob.confirmed.add(applicant)
                    seats[supply_key] -= 1
                    if seats[supply_key] == 0:
                        supply_ob.sold_out = True
                        supply_ob.sellout_capacity = (
                            supply_ob.min_capacity
                            if supply_ob.sellout_capacity is None
                            else supply_ob.sellout_capacity)
            else:
                supply_ob.anomalies.append(f"{ev.event_id}: 零名额下的确认")
            continue

        if ev.event == INVITE:
            ob.invitations += 1
            seen = invited.setdefault(person_key, set())
            if applicant in seen or applicant in waitlist.setdefault(person_key, []):
                ob.duplicate_invites += 1
            else:
                seen.add(applicant)
                waitlist.setdefault(person_key, []).append(applicant)
            continue

        if ev.event == ACCEPT:
            # 接受事件的 request 通常是替代场偏好，候补名单挂在首选场键上
            rank1_slot = _rank1_slot(prefs, applicant, week)
            wl_key = (week, rank1_slot) if rank1_slot else person_key
            wl = waitlist.setdefault(wl_key, [])
            # 流水不保证每次接受都留有前置邀请；不在候补名单时仍按接受入账，
            # 只要求目标场确实有空位
            if applicant in wl:
                wl.remove(applicant)
            if seats.get(supply_key, 0) > 0:
                if applicant not in supply_ob.confirmed:
                    supply_ob.confirmed.add(applicant)
                    seats[supply_key] -= 1
                    if seats[supply_key] == 0:
                        supply_ob.sold_out = True
                # 候补接受另一场：接受型替代（供给槽即实际接受时段）
                if rank1_slot is not None and rank1_slot != ev.slot:
                    flows.append(Flow(applicant, week, rank1_slot, ev.slot,
                                      "accepted_diversion", ev.event_time))
            else:
                supply_ob.anomalies.append(
                    f"{ev.event_id}: 无空位的接受")
            continue

        if ev.event == DECLINE:
            rank1_slot = _rank1_slot(prefs, applicant, week)
            for cand in (person_key,) + (((week, rank1_slot),)
                                         if rank1_slot else ()):
                wl = waitlist.setdefault(cand, [])
                if applicant in wl:
                    wl.remove(applicant)
                    break
            continue

        if ev.event == CANCEL:
            if applicant in supply_ob.confirmed:
                supply_ob.confirmed.remove(applicant)
                seats[supply_key] = seats.get(supply_key, 0) + 1
            wl = waitlist.setdefault(person_key, [])
            if applicant in wl:
                wl.remove(applicant)
            continue

    # 退订回流：撤销 A 之后同周确认/接受 B
    for (applicant, week), timeline in person_events.items():
        timeline.sort()
        cancelled_slot: str | None = None
        cancel_time = -1
        for t, kind, slot in timeline:
            if kind == CANCEL:
                cancelled_slot, cancel_time = slot, t
            elif kind in {CONFIRM, ACCEPT} and cancelled_slot and slot != cancelled_slot \
                    and t > cancel_time:
                flows.append(Flow(applicant, week, cancelled_slot, slot,
                                  "reflow", t))
                cancelled_slot = None

    # 汇总流动与终态
    for f in flows:
        if f.kind == "reflow":
            obs[(f.week, f.to_slot)].reflow_in.add(f.applicant)
        # diverted_in 已由 rank>1 偏好覆盖；accepted_diversion 补进集合
        if f.kind == "accepted_diversion":
            target = obs[(f.week, f.to_slot)]
            if applicant_rank1_is_elsewhere(prefs, f.applicant, f.week, f.to_slot):
                target.diverted_in.add(f.applicant)

    for key, wl in waitlist.items():
        obs[key].final_waitlist = set(wl) - obs[key].confirmed
        # 候补者也可能已在别的场入座（接受替代场），从终态候补剔除
        seated_anywhere = set()
        for other in obs.values():
            seated_anywhere |= other.confirmed
        obs[key].final_waitlist -= seated_anywhere

    for key, ob in obs.items():
        if key not in session_keys and not ob.capacity_known:
            # 没有任何场次也没有名额流水：仅当存在凭据需求时保留，否则标记
            if not ob.credentialed_applicants:
                ob.no_event_records = True
        if not ob.capacity_known and not ob.closed:
            ob.capacity_unknown = True

    return Reconstruction(
        observations=obs,
        flows=flows,
        weeks=weeks,
        unassigned_anonymous_events=unassigned,
        unknown_event_types=unknown_types,
    )


def _rank1_slot(
    prefs: dict[str, dict[str, list[tuple[int, str, str]]]],
    applicant: str,
    week: str,
) -> str | None:
    entries = prefs.get(applicant, {}).get(week)
    if not entries:
        return None
    return sorted(entries)[0][1]


def applicant_rank1_is_elsewhere(
    prefs: dict[str, dict[str, list[tuple[int, str, str]]]],
    applicant: str,
    week: str,
    slot: str,
) -> bool:
    r1 = _rank1_slot(prefs, applicant, week)
    return r1 is not None and r1 != slot
