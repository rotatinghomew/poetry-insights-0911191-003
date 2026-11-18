"""偏好模型、数据加载与 CLI 的测试。"""

import json
import tempfile
import unittest
from pathlib import Path

from workshop.cli import main
from workshop.dataio import (
    Plan, PlanSession, load_plan, load_requests,
)
from workshop.prefs import build_preference_model
from workshop.reconstruct import Flow, Reconstruction, SlotWeekObs


def _recon_with_flow(flow):
    ob = SlotWeekObs(week="W1", slot="周六午")
    return Reconstruction(
        observations={("W1", "周六午"): ob},
        flows=[flow], weeks=["W1"],
        unassigned_anonymous_events=[], unknown_event_types=[])


class PreferenceTests(unittest.TestCase):
    def test_rank1_counts_and_transitions(self):
        requests = []
        for i in range(5):
            requests.append(type("R", (), {
                "request_id": f"a{i}", "applicant": f"P{i}",
                "week": "W1", "slot": "周六早", "rank": 1,
                "origin": "", "visible_at": None})())
            requests.append(type("R", (), {
                "request_id": f"b{i}", "applicant": f"P{i}",
                "week": "W1", "slot": "周六午", "rank": 2,
                "origin": "", "visible_at": None})())
        model = build_preference_model(requests, None)
        self.assertEqual(model.rank1_counts["周六早"], 5)
        self.assertEqual(model.transitions["周六早"]["周六午"], 5)

    def test_behavior_flows_feed_transitions(self):
        requests = [
            type("R", (), {"request_id": "a", "applicant": "P9",
                           "week": "W1", "slot": "周六早", "rank": 1,
                           "origin": "", "visible_at": None})(),
        ]
        flow = Flow(applicant="P9", week="W1", from_slot="周六早",
                    to_slot="周日早", kind="accepted_diversion",
                    event_time=5)
        model = build_preference_model(requests, _recon_with_flow(flow))
        self.assertEqual(model.transitions["周六早"]["周日早"], 1)

    def test_sampled_order_contains_all_slots_once(self):
        import random
        requests = [
            type("R", (), {"request_id": "a", "applicant": "P1",
                           "week": "W1", "slot": "周六早", "rank": 1,
                           "origin": "", "visible_at": None})(),
            type("R", (), {"request_id": "b", "applicant": "P1",
                           "week": "W1", "slot": "周六午", "rank": 2,
                           "origin": "", "visible_at": None})(),
        ]
        model = build_preference_model(requests, None)
        order = model.sample_preference_order("周六早", random.Random(3))
        self.assertEqual(order[0], "周六早")
        self.assertEqual(len(order), len(set(order)))
        self.assertIn("周六午", order)


class DataIOTests(unittest.TestCase):
    def test_plan_marks_added_sessions(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "plan.json"
            p.write_text(json.dumps({
                "name": "t",
                "existing": [{"session_id": "E1", "slot": "周六早",
                              "teacher": "T", "room": "R",
                              "capacity": 10, "minimum": 2}],
                "added": [{"session_id": "N1", "slot": "周六午",
                           "teacher": "T2", "room": "R2",
                           "capacity": 8, "minimum": 3,
                           "tags": ["节庆"]}],
            }), encoding="utf-8")
            plan = load_plan(p)
            self.assertFalse(plan.existing[0].is_added)
            self.assertTrue(plan.added[0].is_added)
            self.assertEqual(plan.added[0].tags, ["节庆"])
            self.assertEqual(len(plan.all_sessions), 2)

    def test_empty_fields_become_none_not_zero(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "requests.csv"
            p.write_text(
                "request_id,applicant,week,preferred_slot,rank,origin,visible_at\n"
                "R1,P1,W1,周六早,1,,\n", encoding="utf-8")
            rows = load_requests(p)
            self.assertEqual(rows[0].origin, "")
            self.assertIsNone(rows[0].visible_at)


class CliSmokeTests(unittest.TestCase):
    def test_all_subcommands_run_on_original_fixture(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(
            main(["classify", "--data-dir", str(root / "data")]), 0)
        self.assertEqual(
            main(["demand", "--data-dir", str(root / "data")]), 0)
        self.assertEqual(
            main(["backtest", "--data-dir", str(root / "data"),
                  "--as-of", "999"]), 0)

    def test_plans_command_reports_conflict(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            plan_dir = Path(td)
            (plan_dir / "bad.json").write_text(json.dumps({
                "name": "bad",
                "existing": [{"session_id": "A", "slot": "周六早",
                              "teacher": "T1", "room": "R1",
                              "capacity": 10, "minimum": 2}],
                "added": [{"session_id": "B", "slot": "周六早",
                           "teacher": "T1", "room": "R2",
                           "capacity": 10, "minimum": 2}],
            }), encoding="utf-8")
            self.assertEqual(
                main(["plans", "--data-dir", str(root / "data"),
                      "--plan-dir", str(plan_dir), "--runs", "50"]), 0)


if __name__ == "__main__":
    unittest.main()
