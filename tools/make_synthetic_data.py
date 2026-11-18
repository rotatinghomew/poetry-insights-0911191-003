"""生成多周合成资料（确定性、可复现），覆盖全部需求分类情形。

用法：
    python -I tools/make_synthetic_data.py example_data

产出到 example_data/ 下：requests.csv / booking_events.csv / sessions.csv。
流程：每周先生成全部潜在申请人（含匿名者），再按统一时间线分配座位，
保证确认/接受事件都发生在真实有空位的场次，候补与售罄浏览只在满额后出现。
"""

from __future__ import annotations

import csv
import math
import random
import sys
from pathlib import Path

SLOTS = ["周六早", "周六午", "周日早", "周日午"]
# 每周每个时段的真实潜在需求均值（含不会留下任何记录的匿名者）
TRUE_LAMBDA = {"周六早": 14, "周六午": 9, "周日早": 6, "周日午": 4}
WEEKS = [f"W{i}" for i in range(1, 7)]


def _poisson(rng: random.Random, lam: float) -> int:
    l = math.exp(-lam)
    k, p = 0, 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= l:
            return k - 1


def main(out_dir: str) -> int:
    rng = random.Random(20261118)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    requests: list[tuple] = []
    events: list[tuple] = []
    sessions: list[tuple] = []
    rid = 1
    eid = 1
    person_seq = 1
    # 迟到补登记录复用的 request_id
    first_request_id: str | None = None

    for wi, week in enumerate(WEEKS):
        closed_slot = "周日早" if week == "W3" else None      # 闭馆：无观测
        zero_slot = "周日午" if week == "W4" else None        # 名额设零：删失
        capacities = {
            "周六早": 10,
            "周六午": 8,
            "周日早": 6,
            "周日午": 0 if zero_slot else 5,
        }
        for si, slot in enumerate(SLOTS):
            sessions.append((f"S{wi*4+si+1}", week, slot,
                             f"T{(si % 3) + 1}", chr(ord("A") + si % 2), 2,
                             "是" if slot == closed_slot else ""))
            if slot == closed_slot:
                events.append((f"E{eid}", "", "闭馆", slot, 0, wi * 100 + 1))
                eid += 1
            else:
                # 显式名额变动：零名额周与满额后的剩余名额 0 由此区分
                events.append((f"E{eid}", "", "名额变动", slot,
                               capacities[slot], wi * 100 + 1, None))
                eid += 1

        # 1) 生成本周潜在申请人及其偏好
        people: list[dict] = []
        for slot in SLOTS:
            if slot == closed_slot:
                continue
            for _ in range(_poisson(rng, TRUE_LAMBDA[slot])):
                person = f"P{person_seq:03d}"
                person_seq += 1
                anonymous = rng.random() < 0.15
                alt = None
                if rng.random() < 0.3:
                    alt = "周六午" if slot == "周六早" else (
                        "周六早" if slot == "周六午" else None)
                if alt == closed_slot:
                    alt = None
                entry = {"person": person, "rank1": slot, "alt": alt,
                         "anonymous": anonymous, "rids": {}}
                if not anonymous:
                    requests.append((f"R{rid}", person, week, slot, 1, "公众号", None))
                    entry["rids"][slot] = f"R{rid}"
                    if first_request_id is None:
                        first_request_id = f"R{rid}"
                    rid += 1
                    if alt:
                        requests.append((f"R{rid}", person, week, alt, 2,
                                         "公众号", None))
                        entry["rids"][alt] = f"R{rid}"
                        rid += 1
                people.append(entry)
        rng.shuffle(people)  # 统一时间线上的到达顺序

        # 2) 时间线分配
        seats = dict(capacities)
        waitlist: dict[str, list[dict]] = {s: [] for s in SLOTS}
        invited: dict[str, set[str]] = {s: set() for s in SLOTS}
        time_cursor = wi * 100 + 2

        for entry in people:
            slot, alt, anon = entry["rank1"], entry["alt"], entry["anonymous"]
            time_cursor += 1
            if seats.get(slot, 0) == 0:
                # 满额（含名额设零）
                if anon:
                    events.append((f"E{eid}", "", "售罄浏览", slot, 0, time_cursor))
                    eid += 1
                    continue
                events.append((f"E{eid}", entry["rids"][slot], "候补邀请",
                               slot, 0, time_cursor))
                eid += 1
                waitlist[slot].append(entry)
                invited[slot].add(entry["person"])
                if rng.random() < 0.2:  # 重复通知
                    time_cursor += 1
                    events.append((f"E{eid}", entry["rids"][slot], "候补邀请",
                                   slot, 0, time_cursor))
                    eid += 1
                roll = rng.random()
                time_cursor += 1
                if roll < 0.45 and alt and seats.get(alt, 0) > 0:
                    seats[alt] -= 1
                    events.append((f"E{eid}", entry["rids"][alt], "接受",
                                   alt, capacities[alt], time_cursor))
                    eid += 1
                    waitlist[slot].remove(entry)
                elif roll < 0.7:
                    events.append((f"E{eid}", entry["rids"][slot], "拒绝",
                                   slot, 0, time_cursor))
                    eid += 1
                    waitlist[slot].remove(entry)
                # 其余留在候补名单
                continue

            # 有名额
            if anon:
                continue  # 匿名者未满额不留痕
            seats[slot] -= 1
            events.append((f"E{eid}", entry["rids"][slot], "确认", slot,
                           capacities[slot], time_cursor))
            eid += 1
            if rng.random() < 0.08:  # 撤销
                time_cursor += 1
                events.append((f"E{eid}", entry["rids"][slot], "撤销", slot,
                               capacities[slot], time_cursor))
                eid += 1
                seats[slot] += 1
                if alt and rng.random() < 0.6 and seats.get(alt, 0) > 0:
                    seats[alt] -= 1
                    time_cursor += 1
                    events.append((f"E{eid}", entry["rids"][alt], "确认", alt,
                                   capacities[alt], time_cursor))
                    eid += 1

        # 3) 出席/缺席事件（永远不进入预测输入）
        confirm_events = [row for row in events
                          if row[2] == "确认" and row[5] > wi * 100
                          and row[5] < (wi + 1) * 100]
        for row in confirm_events:
            if rng.random() < 0.9:
                events.append((f"E{eid}", row[1], "出席", row[3], row[4],
                               row[5] + 1, None))
                eid += 1
            else:
                events.append((f"E{eid}", row[1], "缺席", row[3], row[4],
                               row[5] + 1, None))
                eid += 1

    # 事后补登：recorded_at 远晚于事件时点，回测中会被剔除
    events.append(("E9001", first_request_id, "拒绝", "周六早", 0, 150, 560))

    _write(out / "requests.csv",
           ["request_id", "applicant", "week", "preferred_slot", "rank",
            "origin", "visible_at"], requests)
    events_with_week = []
    for row in events:
        row = tuple(row) + (None,) * (7 - len(row))  # 补齐 recorded_at 槽位
        if row[0] == "E9001":
            wk = "W2"  # 迟到补登：事件发生在历史周，但 recorded_at 很晩
        else:
            wk = WEEKS[min(len(WEEKS) - 1, row[5] // 100)]
        events_with_week.append(row + (wk,))
    _write(out / "booking_events.csv",
           ["event_id", "request_id", "event", "slot", "capacity",
            "event_time", "recorded_at", "week"], events_with_week)
    _write(out / "sessions.csv",
           ["session_id", "week", "slot", "teacher", "room", "minimum",
            "closed"], sessions)
    print(f"wrote {len(requests)} requests, {len(events)} events, "
          f"{len(sessions)} sessions to {out}")
    return 0


def _write(path: Path, header: list[str], rows: list[tuple]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for row in rows:
            writer.writerow(["" if v is None else v for v in row])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -I tools/make_synthetic_data.py OUT_DIR")
    raise SystemExit(main(sys.argv[1]))
