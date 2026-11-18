"""数据加载层：三个 CSV（可带可选列）与增场方案 JSON。

CSV 均为 UTF-8，空字段表示该记录未提供，而不是“没有发生”。
所有时间字段以可比较的整数/字符串读入，事件按 event_time 排序重建。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---- 原始记录 ---------------------------------------------------------------

@dataclass(frozen=True)
class Request:
    """一条带凭据的首次需求记录（一个申请人在一周内的一个可接受时段）。"""

    request_id: str
    applicant: str
    week: str
    slot: str
    rank: int
    origin: str = ""
    visible_at: int | None = None  # 可选：该条记录在当时排期系统可见的时点


@dataclass(frozen=True)
class BookingEvent:
    """预约流水中的一条事件。

    event 取值：确认 / 候补邀请 / 接受 / 拒绝 / 撤销 / 售罄浏览 / 闭馆 / 出席 / 缺席
    - capacity 为该事件时点该时段对外可见的名额；空为未知。
    - recorded_at 为可选列：事后补登事件的补登时点，用于点-in-time 回测。
    """

    event_id: str
    request_id: str
    event: str
    slot: str
    capacity: int | None
    event_time: int
    recorded_at: int | None = None
    week: str | None = None  # 可选列：匿名供给事件（名额变动/闭馆）的归属周


@dataclass(frozen=True)
class Session:
    """实际排过的一场工坊。"""

    session_id: str
    week: str
    slot: str
    teacher: str
    room: str
    minimum: int
    closed: bool = False  # 可选列：该场是否闭馆（与名额设零不同）


@dataclass
class PlanSession:
    """候选方案中的一场（未实际办过也允许）。"""

    session_id: str
    slot: str
    teacher: str
    room: str
    capacity: int
    minimum: int
    acceptable_slots: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)  # 如 节庆 / 新品类
    is_added: bool = False  # 由 Plan 装载时标记：新增场为 True


@dataclass
class Plan:
    """一套可实际执行的下月增场方案。

    existing 为下月保留的旧场（capacity 为其下月名额），
    added 为新增场；两者共同构成联合供给，需求在整盘内分流。
    """

    name: str
    existing: list[PlanSession]
    added: list[PlanSession]

    @property
    def all_sessions(self) -> list[PlanSession]:
        return self.existing + self.added


# ---- CSV 读取 ---------------------------------------------------------------

def _open_csv(path: str | Path):
    return Path(path).open(encoding="utf-8", newline="")


def _int_or_none(value: str) -> int | None:
    value = value.strip()
    if value == "":
        return None
    return int(value)


def load_requests(path: str | Path) -> list[Request]:
    out: list[Request] = []
    with _open_csv(path) as fh:
        for row in csv.DictReader(fh):
            out.append(
                Request(
                    request_id=row["request_id"].strip(),
                    applicant=row["applicant"].strip(),
                    week=row["week"].strip(),
                    slot=row["preferred_slot"].strip(),
                    rank=int(row["rank"]),
                    origin=(row.get("origin") or "").strip(),
                    visible_at=_int_or_none(row.get("visible_at", "") or ""),
                )
            )
    return out


def load_events(path: str | Path) -> list[BookingEvent]:
    out: list[BookingEvent] = []
    with _open_csv(path) as fh:
        for row in csv.DictReader(fh):
            out.append(
                BookingEvent(
                    event_id=row["event_id"].strip(),
                    request_id=(row.get("request_id") or "").strip(),
                    event=row["event"].strip(),
                    slot=row["slot"].strip(),
                    capacity=_int_or_none(row.get("capacity", "") or ""),
                    event_time=int(row["event_time"]),
                    recorded_at=_int_or_none(row.get("recorded_at", "") or ""),
                    week=(row.get("week") or "").strip() or None,
                )
            )
    out.sort(key=lambda e: (e.event_time, e.event_id))
    return out


def load_sessions(path: str | Path) -> list[Session]:
    out: list[Session] = []
    with _open_csv(path) as fh:
        for row in csv.DictReader(fh):
            closed_raw = (row.get("closed") or "").strip()
            out.append(
                Session(
                    session_id=row["session_id"].strip(),
                    week=row["week"].strip(),
                    slot=row["slot"].strip(),
                    teacher=row["teacher"].strip(),
                    room=row["room"].strip(),
                    minimum=int(row["minimum"]),
                    closed=closed_raw in {"1", "true", "True", "是", "闭馆"},
                )
            )
    return out


# ---- 方案 JSON --------------------------------------------------------------

def _plan_session(raw: dict[str, Any]) -> PlanSession:
    return PlanSession(
        session_id=str(raw["session_id"]),
        slot=str(raw["slot"]),
        teacher=str(raw["teacher"]),
        room=str(raw["room"]),
        capacity=int(raw["capacity"]),
        minimum=int(raw.get("minimum", 1)),
        acceptable_slots=[str(s) for s in raw.get("acceptable_slots", [])],
        tags=[str(t) for t in raw.get("tags", [])],
    )


def load_plan(path: str | Path) -> Plan:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    existing = [_plan_session(x) for x in raw.get("existing", [])]
    added = [_plan_session(x) for x in raw.get("added", [])]
    for s in added:
        s.is_added = True
    return Plan(
        name=str(raw.get("name", Path(path).stem)),
        existing=existing,
        added=added,
    )
