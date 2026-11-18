"""方案评估测试：联合分流、净新增/挤出、硬约束、最低开课取消。"""

import unittest

from workshop.dataio import Plan, PlanSession, Request
from workshop.demand import SlotDemandEstimate
from workshop.planning import check_constraints, evaluate_plan
from workshop.prefs import build_preference_model
from workshop.reconstruct import Reconstruction, SlotWeekObs


def _est(slot, point, alpha=20.0, beta=2.0):
    return SlotDemandEstimate(
        slot=slot, point=point, ci_low=point * 0.7, ci_high=point * 1.3,
        observed_mean=point, weeks_total=6, uncensored_weeks=4,
        censored_weeks=2, closed_weeks=0, anonymous_full_views=0,
        waitlist_persons=0, zero_capacity_weeks=0,
        capacity_unknown_weeks=0, grade="A",
        grade_reasons=[], posterior_alpha=alpha, posterior_beta=beta)


def _prefs():
    # 每个申请人首选周六早，次选周六午
    requests = []
    for i in range(40):
        requests.append(Request(f"r{i}a", f"P{i:03d}", "W1", "周六早", 1))
        requests.append(Request(f"r{i}b", f"P{i:03d}", "W1", "周六午", 2))
    return build_preference_model(requests, None)


class ConstraintTests(unittest.TestCase):
    def test_teacher_and_room_conflicts_detected(self):
        plan = Plan("x",
                    existing=[PlanSession("A", "周六早", "T1", "R1", 10, 2)],
                    added=[PlanSession("B", "周六早", "T1", "R2", 10, 2),
                           PlanSession("C", "周六早", "T2", "R1", 10, 2)])
        conflicts = check_constraints(plan, {"周六早"})
        kinds = {c.kind for c in conflicts}
        self.assertIn("teacher_conflict", kinds)
        self.assertIn("room_conflict", kinds)
        self.assertFalse(
            evaluate_plan(plan, {}, _prefs(), known_slots={"周六早"}).feasible)

    def test_minimum_above_capacity_rejected(self):
        plan = Plan("x", existing=[],
                    added=[PlanSession("A", "周六早", "T1", "R1", 2, 5)])
        kinds = {c.kind for c in check_constraints(plan, {"周六早"})}
        self.assertIn("minimum_above_capacity", kinds)

    def test_unknown_slot_requires_declaration(self):
        plan = Plan("x", existing=[],
                    added=[PlanSession("A", "全新晚场", "T1", "R1", 8, 2)])
        conflicts = check_constraints(plan, {"周六早"})
        self.assertTrue(any(c.kind == "unknown_slot" for c in conflicts))


class JointEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.estimates = {
            "周六早": _est("周六早", 20.0),
            "周六午": _est("周六午", 8.0),
        }
        self.prefs = _prefs()
        self.known = {"周六早", "周六午"}
        self.base = [PlanSession("OLD-AM", "周六早", "T1", "R1", 10, 2)]

    @staticmethod
    def _plan(existing, added):
        for s in added:
            s.is_added = True
        return Plan("p", existing=existing, added=added)

    def test_served_is_not_sum_of_independent_predictions(self):
        """同一申请人只服务一次：总服务人数 <= 各场容量之和，且受联合分流约束。"""
        plan = self._plan(self.base, [
            PlanSession("NEW-PM", "周六午", "T2", "R2", 10, 2,
                        acceptable_slots=["周六午", "周六早"]),
        ])
        ev = evaluate_plan(plan, self.estimates, self.prefs,
                           n_runs=400, seed=7, known_slots=self.known)
        self.assertTrue(ev.feasible)
        total_capacity = sum(s.capacity for s in plan.all_sessions)
        self.assertLessEqual(ev.expected_served, total_capacity + 0.01)
        # 总需求期望约 28；20 个容量意味着必然有人未被服务
        self.assertGreater(ev.expected_unserved, 0.0)

    def test_added_session_diverts_from_old_but_adds_net_served(self):
        plan = self._plan(self.base, [
            PlanSession("NEW-AM2", "周六早", "T3", "R3", 10, 2,
                        acceptable_slots=["周六早", "周六午"]),
        ])
        ev = evaluate_plan(plan, self.estimates, self.prefs,
                           n_runs=400, seed=7, known_slots=self.known)
        # 旧场优先承接，保留绝大部分入座者；新增场带来净新增服务
        self.assertGreater(ev.newly_served_vs_baseline, 1.0)
        old = next(s for s in ev.per_session if not s.is_added)
        self.assertGreater(old.expected_enrolled, 7.0)
        self.assertLess(ev.displaced_vs_baseline, ev.newly_served_vs_baseline)

    def test_minimum_enrollment_cancels_session_and_recounts(self):
        """高最低开课人数的新增场大概率取消，取消率与不达标概率一致。"""
        plan = self._plan(self.base, [
            PlanSession("NEW-PM", "周六午", "T2", "R2", 10, 9,
                        acceptable_slots=["周六午", "周六早"]),
        ])
        ev = evaluate_plan(plan, self.estimates, self.prefs,
                           n_runs=400, seed=7, known_slots=self.known)
        new = next(s for s in ev.per_session if s.is_added)
        self.assertGreater(new.cancel_rate, 0.2)
        self.assertLess(new.prob_meets_minimum, 0.8)

    def test_grade_c_slot_requires_explicit_assumption(self):
        estimates = dict(self.estimates)
        estimates["节庆晚场"] = SlotDemandEstimate(
            "节庆晚场", point=5.0, ci_low=2.0, ci_high=9.0,
            observed_mean=None, weeks_total=0, uncensored_weeks=0,
            censored_weeks=0, closed_weeks=0, anonymous_full_views=0,
            waitlist_persons=0, zero_capacity_weeks=0,
            capacity_unknown_weeks=0, grade="C",
            grade_reasons=["节庆新条件"], posterior_alpha=5,
            posterior_beta=1)
        plan = self._plan(self.base, [
            PlanSession("FEST", "节庆晚场", "T9", "R9", 8, 3,
                        acceptable_slots=["节庆晚场"]),
        ])
        blocked = evaluate_plan(plan, estimates, self.prefs,
                                n_runs=100, seed=7, known_slots=self.known)
        self.assertTrue(blocked.evidence_gaps)  # 无假设时只报证据缺口
        allowed = evaluate_plan(plan, estimates, self.prefs,
                                n_runs=100, seed=7,
                                assumptions={"节庆晚场": 5.0},
                                known_slots=self.known)
        self.assertTrue(allowed.feasible)
        self.assertEqual(allowed.assumptions, {"节庆晚场": 5.0})

    def test_deterministic_with_same_seed(self):
        plan = self._plan(self.base, [
            PlanSession("NEW-AM2", "周六早", "T3", "R3", 10, 2,
                        acceptable_slots=["周六早", "周六午"]),
        ])
        a = evaluate_plan(plan, self.estimates, self.prefs,
                          n_runs=200, seed=11, known_slots=self.known)
        b = evaluate_plan(plan, self.estimates, self.prefs,
                          n_runs=200, seed=11, known_slots=self.known)
        self.assertEqual(a.expected_served, b.expected_served)
        self.assertEqual(a.newly_served_vs_baseline,
                         b.newly_served_vs_baseline)


if __name__ == "__main__":
    unittest.main()
