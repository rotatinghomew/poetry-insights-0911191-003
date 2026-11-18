"""命令行入口：

python -m workshop classify --data-dir data
python -m workshop demand   --data-dir data [--plan plan.json]
python -m workshop plans    --data-dir data --plan-dir example_data --runs 1000
python -m workshop backtest --data-dir data --as-of 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .backtest import run_backtest
from .dataio import load_events, load_plan, load_requests, load_sessions
from .demand import GRADE_A, GRADE_B, GRADE_C, estimate_demand
from .planning import evaluate_plan
from .prefs import build_preference_model
from .reconstruct import reconstruct


def _load_all(data_dir: Path):
    requests = load_requests(data_dir / "requests.csv")
    events = load_events(data_dir / "booking_events.csv")
    sessions = load_sessions(data_dir / "sessions.csv")
    return requests, events, sessions


def _print(payload: dict) -> None:
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


def cmd_classify(args: argparse.Namespace) -> int:
    requests, events, sessions = _load_all(Path(args.data_dir))
    recon = reconstruct(requests, events, sessions)
    payload = {
        "slot_weeks": [
            {
                "week": ob.week,
                "slot": ob.slot,
                "rank1_unique_demand": ob.observed_demand,
                "credentialed_applicants": sorted(ob.credentialed_applicants),
                "diverted_in": sorted(ob.diverted_in),
                "reflow_in": sorted(ob.reflow_in),
                "invitations": ob.invitations,
                "duplicate_invites": ob.duplicate_invites,
                "sold_out": ob.sold_out,
                "sellout_capacity": ob.sellout_capacity,
                "anonymous_full_views": ob.anonymous_full_views,
                "identified_full_views": ob.identified_full_views,
                "closed": ob.closed,
                "zero_capacity": ob.zero_capacity,
                "capacity_unknown": ob.capacity_unknown,
                "no_event_records": ob.no_event_records,
                "censored": ob.censored,
                "final_waitlist": sorted(ob.final_waitlist),
                "anomalies": ob.anomalies,
            }
            for ob in sorted(recon.observations.values(),
                             key=lambda o: (o.week, o.slot))
        ],
        "flows": [f.__dict__ for f in recon.flows],
        "unassigned_anonymous_events": recon.unassigned_anonymous_events,
        "unknown_event_types": recon.unknown_event_types,
    }
    _print(payload)
    return 0


def _plan_tags(plan_path: str | None) -> dict[str, list[str]]:
    if not plan_path:
        return {}
    plan = load_plan(plan_path)
    return {s.slot: s.tags for s in plan.all_sessions if s.tags}


def cmd_demand(args: argparse.Namespace) -> int:
    requests, events, sessions = _load_all(Path(args.data_dir))
    recon = reconstruct(requests, events, sessions)
    estimates = estimate_demand(recon, _plan_tags(args.plan))
    payload = {
        "slot_estimates": [e.as_dict() for _, e in sorted(estimates.items())],
        "evidence_legend": {
            GRADE_A: "资料较足，可直接用于下月排期",
            GRADE_B: "有历史支持但删失/样本有限，区间较宽，需保守使用",
            GRADE_C: "不能从历史支持（无观测/仅匿名/新品类或节庆），"
                     "应先小范围试开并显式声明需求假设",
        },
    }
    _print(payload)
    return 0


def cmd_plans(args: argparse.Namespace) -> int:
    requests, events, sessions = _load_all(Path(args.data_dir))
    recon = reconstruct(requests, events, sessions)
    estimates = estimate_demand(recon)
    prefs = build_preference_model(requests, recon)
    known_slots = set(estimates)

    plan_paths = sorted(Path(args.plan_dir).glob("*.json"))
    if not plan_paths:
        print(f"目录 {args.plan_dir} 下没有方案 JSON", file=sys.stderr)
        return 2

    reports = []
    for path in plan_paths:
        plan = load_plan(path)
        assumptions = json.loads(path.read_text(encoding="utf-8")).get(
            "demand_assumptions", {})
        show_rates = json.loads(path.read_text(encoding="utf-8")).get(
            "show_rates", {})
        ev = evaluate_plan(
            plan, estimates, prefs,
            n_runs=args.runs, seed=args.seed,
            assumptions=assumptions, show_rates=show_rates,
            known_slots=known_slots)
        reports.append(ev.as_dict())
    _print({"plans": reports})
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    requests, events, sessions = _load_all(Path(args.data_dir))
    report = run_backtest(requests, events, sessions, as_of=args.as_of,
                          eval_weeks=args.eval_weeks or None)
    _print(report.as_dict())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="workshop",
                                     description="唐诗工坊需求估计与增场评估")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("classify", help="重建事件流并输出需求分类")
    p.add_argument("--data-dir", required=True)
    p.set_defaults(func=cmd_classify)

    p = sub.add_parser("demand", help="逐时段潜在需求估计与证据分级")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--plan", help="可选方案 JSON，用于识别新品类/节庆标签")
    p.set_defaults(func=cmd_demand)

    p = sub.add_parser("plans", help="联合模拟评估多套增场方案")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--plan-dir", required=True)
    p.add_argument("--runs", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20261118)
    p.set_defaults(func=cmd_plans)

    p = sub.add_parser("backtest", help="点-in-time 回测实际办过的安排")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--as-of", type=int, required=True)
    p.add_argument("--eval-weeks", nargs="*")
    p.set_defaults(func=cmd_backtest)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
