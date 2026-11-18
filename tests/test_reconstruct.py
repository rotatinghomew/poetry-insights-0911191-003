"""需求分类的陷阱测试：转场、回流、重复通知、匿名满额、闭馆 vs 零名额。"""

import unittest

from workshop.dataio import BookingEvent as E
from workshop.dataio import Request as R
from workshop.dataio import Session as S
from workshop.reconstruct import (
    reconstruct,
)


def req(rid, person, week, slot, rank):
    return R(rid, person, week, slot, rank, "公众号", None)


def ev(eid, rid, kind, slot, cap, t, week=None, recorded_at=None):
    return E(eid, rid, kind, slot, cap, t, recorded_at, week)


class ClassificationTests(unittest.TestCase):
    def setUp(self):
        # 一个满额场（周六早，2 名额）+ 一个替代场（周六午）
        self.requests = [
            req("R01", "P01", "W1", "周六早", 1),
            req("R02", "P01", "W1", "周六午", 2),
            req("R03", "P02", "W1", "周六早", 1),
            req("R04", "P03", "W1", "周六早", 1),
            req("R05", "P03", "W1", "周六午", 2),
            req("R06", "P04", "W1", "周六午", 1),
        ]
        self.sessions = [
            S("S1", "W1", "周六早", "T01", "A", 2),
            S("S2", "W1", "周六午", "T02", "B", 2),
        ]

    def test_diversion_is_not_new_demand(self):
        """rank>1 偏好与候补接受另一场都不能算新增独立需求。"""
        events = [
            ev("B01", "", "名额变动", "周六早", 2, 1, week="W1"),
            ev("B02", "R01", "确认", "周六早", 2, 2),
            ev("B03", "R03", "确认", "周六早", 2, 3),
            # 满额后 P03 收到候补邀请（R04 是 P03 的首选请求），
            # 随后用替代场请求 R05 接受周六午；P04 收到重复邀请
            ev("B04", "R04", "候补邀请", "周六早", 0, 4),
            ev("B05", "R04", "候补邀请", "周六早", 0, 5),
            ev("B06", "R05", "接受", "周六午", 2, 6),
        ]
        recon = reconstruct(self.requests, events, self.sessions)
        am = recon.obs("W1", "周六早")
        pm = recon.obs("W1", "周六午")
        self.assertEqual(am.observed_demand, 3)   # P01/P02/P03 三个独立首选
        self.assertEqual(pm.observed_demand, 1)   # 只有 P04 是首选；P03 不算新增
        self.assertIn("P03", pm.diverted_in)
        kinds = {f.kind for f in recon.flows}
        self.assertIn("accepted_diversion", kinds)
        self.assertIn("preference_diversion", kinds)

    def test_cancel_reflow_is_not_new_demand(self):
        """撤销 A 后确认 B 是回流，B 的独立需求不增加。"""
        events = [
            ev("B01", "", "名额变动", "周六早", 2, 1, week="W1"),
            ev("B02", "", "名额变动", "周六午", 2, 1, week="W1"),
            ev("B03", "R01", "确认", "周六早", 2, 2),
            ev("B04", "R03", "确认", "周六早", 2, 3),
            ev("B05", "R01", "撤销", "周六早", 2, 4),
            ev("B06", "R02", "确认", "周六午", 2, 5),
            ev("B07", "R06", "确认", "周六午", 2, 6),
        ]
        recon = reconstruct(self.requests, events, self.sessions)
        reflows = [f for f in recon.flows if f.kind == "reflow"]
        self.assertEqual(len(reflows), 1)
        self.assertEqual(reflows[0].from_slot, "周六早")
        self.assertEqual(reflows[0].to_slot, "周六午")
        pm = recon.obs("W1", "周六午")
        self.assertEqual(pm.observed_demand, 1)  # 仍是 P04，P01 回流不计新增
        self.assertIn("P01", pm.reflow_in)

    def test_duplicate_invitations_count_once(self):
        """同一申请人的重复候补邀请只计一次人。"""
        events = [
            ev("B01", "", "名额变动", "周六早", 1, 1, week="W1"),
            ev("B02", "R01", "确认", "周六早", 1, 2),
            ev("B03", "R03", "候补邀请", "周六早", 0, 3),
            ev("B04", "R03", "候补邀请", "周六早", 0, 4),  # 重复
            ev("B05", "R04", "候补邀请", "周六早", 0, 5),
        ]
        recon = reconstruct(self.requests, events, self.sessions)
        am = recon.obs("W1", "周六早")
        self.assertEqual(am.invitations, 3)
        self.assertEqual(am.duplicate_invites, 1)
        self.assertEqual(len(am.final_waitlist), 2)  # P03、P04 各一次

    def test_anonymous_full_view_is_evidence_not_zero(self):
        """匿名满额浏览证明存在未观测需求，但不能计入人数。"""
        events = [
            ev("B01", "", "名额变动", "周六早", 1, 1, week="W1"),
            ev("B02", "R01", "确认", "周六早", 1, 2),
            ev("B03", "", "售罄浏览", "周六早", 0, 3, week="W1"),
            ev("B04", "", "售罄浏览", "周六早", 0, 4, week="W1"),
        ]
        recon = reconstruct(self.requests, events, self.sessions)
        am = recon.obs("W1", "周六早")
        self.assertTrue(am.sold_out)
        self.assertEqual(am.anonymous_full_views, 2)
        self.assertEqual(am.observed_demand, 3)  # 匿名者不计入
        # 没有身份的事件不会被凭空归到别的 slot-week
        self.assertEqual(recon.unassigned_anonymous_events, [])

    def test_closure_vs_zero_capacity_are_different(self):
        """闭馆=无观测（不删失）；名额设零=右删失观测。"""
        requests = [req("R10", "P10", "W1", "周日早", 1),
                    req("R11", "P11", "W1", "周日午", 1)]
        sessions = [
            S("S3", "W1", "周日早", "T03", "C", 2, closed=True),
            S("S4", "W1", "周日午", "T04", "D", 2),
        ]
        events = [
            ev("C01", "", "闭馆", "周日早", 0, 1, week="W1"),
            ev("C02", "", "名额变动", "周日午", 0, 1, week="W1"),
            ev("C03", "R11", "售罄浏览", "周日午", 0, 2),
        ]
        recon = reconstruct(requests, events, sessions)
        closed = recon.obs("W1", "周日早")
        zeroed = recon.obs("W1", "周日午")
        self.assertTrue(closed.closed)
        self.assertFalse(closed.censored)       # 闭馆不删失
        self.assertFalse(closed.zero_capacity)
        self.assertTrue(zeroed.zero_capacity)
        self.assertTrue(zeroed.censored)        # 零名额是删失
        self.assertFalse(zeroed.closed)

    def test_sellout_marks_censoring_with_capacity_threshold(self):
        """满额后该 slot-week 标记删失，删失阈值等于生效名额。"""
        events = [
            ev("B01", "", "名额变动", "周六早", 2, 1, week="W1"),
            ev("B02", "R01", "确认", "周六早", 2, 2),
            ev("B03", "R03", "确认", "周六早", 2, 3),
            ev("B04", "R04", "候补邀请", "周六早", 0, 4),
        ]
        recon = reconstruct(self.requests, events, self.sessions)
        am = recon.obs("W1", "周六早")
        self.assertTrue(am.sold_out)
        self.assertTrue(am.censored)
        self.assertEqual(am.sellout_capacity, 2)

    def test_original_fixture_runs_without_anomaly(self):
        """仓库自带小样本（含满额浏览与候补）必须无异常重建。"""
        from pathlib import Path
        from workshop.dataio import (
            load_events, load_requests, load_sessions,
        )
        root = Path(__file__).resolve().parents[1] / "data"
        rq = load_requests(root / "requests.csv")
        evs = load_events(root / "booking_events.csv")
        ss = load_sessions(root / "sessions.csv")
        recon = reconstruct(rq, evs, ss)
        anomalies = [a for o in recon.observations.values() for a in o.anomalies]
        self.assertEqual(anomalies, [])
        am = recon.obs("W1", "周六早")
        self.assertTrue(am.sold_out)


if __name__ == "__main__":
    unittest.main()
