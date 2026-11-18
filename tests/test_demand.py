"""潜在需求估计测试：删失修正方向、闭馆不入似然、分级与新品类降级。"""

import math
import unittest

from workshop.dataio import BookingEvent as E
from workshop.dataio import Request as R
from workshop.dataio import Session as S
from workshop.demand import (
    GRADE_A, GRADE_B, GRADE_C, estimate_demand,
)
from workshop.reconstruct import reconstruct


def _week_records(week, slot, persons, cap, sellout_after=None,
                  closed=False, zero=False):
    requests = [R(f"{week}-{slot}-{p}", p, week, slot, 1, "x", None)
                for p in persons]
    events = []
    eid = 1
    if closed:
        events.append(E(f"{week}{eid}", "", "闭馆", slot, 0, 1, None, week))
        return requests, events
    events.append(E(f"{week}{eid}", "", "名额变动", slot,
                    0 if zero else cap, 1, None, week))
    eid += 1
    if zero:
        for p in persons:
            events.append(E(f"{week}{eid}", f"{week}-{slot}-{p}",
                            "售罄浏览", slot, 0, eid, None, week))
            eid += 1
        return requests, events
    for i, p in enumerate(persons):
        if sellout_after is not None and i >= sellout_after:
            events.append(E(f"{week}{eid}", f"{week}-{slot}-{p}",
                            "候补邀请", slot, 0, eid, None, week))
        else:
            events.append(E(f"{week}{eid}", f"{week}-{slot}-{p}",
                            "确认", slot, cap, eid, None, week))
        eid += 1
    return requests, events


class DemandTests(unittest.TestCase):
    def _build(self, per_week):
        requests, events, sessions = [], [], []
        for i, (n, sellout, closed, zero) in enumerate(per_week):
            week = f"W{i+1}"
            persons = [f"P{week}{j}" for j in range(n)]
            sessions.append(S(f"S{week}", week, "X", "T", "A", 2, closed))
            rq, ev = _week_records(week, "X", persons, cap=5,
                                   sellout_after=5 if sellout else None,
                                   closed=closed, zero=zero)
            requests += rq
            events += ev
        recon = reconstruct(requests, events, sessions)
        return recon

    def test_censored_estimate_exceeds_observed_mean(self):
        """满额周的真实需求被截断，EM 估计必须高于观测均值。"""
        # 6 周全观测到 5 人且满额：真实 D>=5，估计应明显高于 5 的观测口径
        recon = self._build([(5, True, False, False)] * 6)
        est = estimate_demand(recon)["X"]
        observed = [o.observed_demand for o in recon.slot_weeks("X")]
        self.assertTrue(all(v == 5 for v in observed))
        self.assertGreater(est.point, 5.0)
        self.assertEqual(est.censored_weeks, 6)
        self.assertLessEqual(est.ci_low, est.point)
        self.assertGreaterEqual(est.ci_high, est.point)

    def test_closed_weeks_are_not_observations(self):
        """闭馆周不进似然：估计只基于其余周，且闭馆周被单列。"""
        recon = self._build([
            (3, False, False, False), (3, False, False, False),
            (0, False, True, False),  # 闭馆
            (3, False, False, False),
        ])
        est = estimate_demand(recon)["X"]
        self.assertEqual(est.closed_weeks, 1)
        self.assertEqual(est.weeks_total, 4)
        self.assertAlmostEqual(est.observed_mean, 3.0)
        self.assertTrue(any("闭馆" in r for r in est.grade_reasons))

    def test_zero_capacity_is_censored_not_closure(self):
        recon = self._build([
            (4, False, False, False), (4, False, False, False),
            (4, False, False, True),   # 名额设零
        ])
        est = estimate_demand(recon)["X"]
        self.assertEqual(est.zero_capacity_weeks, 1)
        self.assertEqual(est.censored_weeks, 1)
        self.assertEqual(est.closed_weeks, 0)
        self.assertTrue(any("名额设零" in r for r in est.grade_reasons))

    def test_all_censored_slot_gets_grade_c(self):
        recon = self._build([(5, True, False, False)] * 3)
        est = estimate_demand(recon)["X"]
        self.assertEqual(est.uncensored_weeks, 0)
        self.assertEqual(est.grade, GRADE_C)

    def test_uncensored_history_gets_grade_a(self):
        recon = self._build([(2, False, False, False)] * 4)
        est = estimate_demand(recon)["X"]
        self.assertEqual(est.grade, GRADE_A)
        self.assertGreaterEqual(est.uncensored_weeks, 3)

    def test_mixed_history_gets_grade_b(self):
        recon = self._build([
            (3, False, False, False), (5, True, False, False),
            (5, True, False, False),
        ])
        est = estimate_demand(recon)["X"]
        self.assertEqual(est.grade, GRADE_B)

    def test_new_festival_tag_forces_grade_c(self):
        recon = self._build([(2, False, False, False)] * 4)
        plain = estimate_demand(recon)["X"]
        tagged = estimate_demand(recon, {"X": ["节庆"]})["X"]
        self.assertEqual(plain.grade, GRADE_A)
        self.assertEqual(tagged.grade, GRADE_C)
        self.assertTrue(any("节庆" in r for r in tagged.grade_reasons))

    def test_no_history_slot_has_no_point_estimate(self):
        # 只有闭馆、没有任何需求观测的时段
        recon = self._build([(0, False, True, False)])
        est = estimate_demand(recon)["X"]
        self.assertIsNone(est.point)
        self.assertEqual(est.grade, GRADE_C)

    def test_poisson_em_math_identity(self):
        from workshop.demand import _censored_effective, _poisson_cdf
        lam = 3.0
        # c=0 时无条件，期望就是 λ
        self.assertAlmostEqual(_censored_effective(lam, 0), lam)
        # E[D|D>=c] >= c 且随 c 增大
        vals = [_censored_effective(lam, c) for c in range(1, 6)]
        for c, v in enumerate(vals, start=1):
            self.assertGreaterEqual(v, c - 1e-9)
        self.assertTrue(all(vals[i] <= vals[i+1] for i in range(len(vals)-1)))
        self.assertTrue(0.0 < _poisson_cdf(2, lam) < 1.0)


if __name__ == "__main__":
    unittest.main()
