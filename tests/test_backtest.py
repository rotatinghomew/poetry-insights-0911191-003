"""点-in-time 回测测试：补登泄漏、出席不入预测、周窗口纪律。"""

import unittest

from workshop.backtest import run_backtest
from workshop.dataio import BookingEvent as E
from workshop.dataio import Request as R
from workshop.dataio import Session as S


def _ev(eid, rid, kind, slot, cap, t, week, recorded_at=None):
    return E(eid, rid, kind, slot, cap, t, recorded_at, week)


class BacktestTests(unittest.TestCase):
    def _data(self):
        # W1/W2 低需求未删失；W3 高需求满额
        requests = []
        for w, n in (("W1", 2), ("W2", 2), ("W3", 9)):
            for j in range(n):
                requests.append(R(f"{w}-r{j}", f"{w}-P{j}", w, "X", 1,
                                  "x", None))
        events = [
            _ev("e1", "", "名额变动", "X", 5, 101, "W1"),
            _ev("e2", "W1-r0", "确认", "X", 5, 102, "W1"),
            _ev("e3", "W1-r1", "确认", "X", 5, 103, "W1"),
            _ev("e4", "", "名额变动", "X", 5, 201, "W2"),
            _ev("e5", "W2-r0", "确认", "X", 5, 202, "W2"),
            _ev("e6", "W2-r1", "确认", "X", 5, 203, "W2"),
            _ev("e7", "", "名额变动", "X", 5, 301, "W3"),
        ]
        for j in range(9):
            kind = "确认" if j < 5 else "候补邀请"
            cap = 5 if j < 5 else 0
            events.append(_ev(f"w3-{j}", f"W3-r{j}", kind, "X", cap,
                              302 + j, "W3"))
        # W3 的出席事件：永不进入预测输入
        for j in range(5):
            events.append(_ev(f"att{j}", f"W3-r{j}", "出席", "X", 5,
                              340 + j, "W3"))
        sessions = [S(f"S{w}", w, "X", "T", "A", 2)
                    for w in ("W1", "W2", "W3")]
        return requests, events, sessions

    def test_train_eval_window_split(self):
        requests, events, sessions = self._data()
        rep = run_backtest(requests, events, sessions, as_of=299)
        self.assertEqual(rep.train_weeks, ["W1", "W2"])
        self.assertEqual(rep.eval_weeks, ["W3"])

    def test_late_backfilled_response_excluded_from_prediction(self):
        requests, events, sessions = self._data()
        # W2 的一条拒绝，事后（as_of=299 之后）才补登
        events.append(_ev("late1", "W2-r1", "拒绝", "X", 0, 205, "W2",
                          recorded_at=350))
        rep = run_backtest(requests, events, sessions, as_of=299)
        self.assertIn("late1", rep.excluded_late_events)

    def test_attendance_never_enters_prediction_input(self):
        requests, events, sessions = self._data()
        rep = run_backtest(requests, events, sessions, as_of=299)
        # 出席事件不出现在被剔除清单（它们走单独的永久排除通道），
        # 但实际值里能看到出席数
        slot = next(s for s in rep.slots if s.slot == "X")
        self.assertEqual(slot.actual_attendance, 5)
        self.assertNotIn("att0", rep.excluded_late_events)

    def test_censored_eval_week_scored_by_lower_bound(self):
        requests, events, sessions = self._data()
        rep = run_backtest(requests, events, sessions, as_of=299)
        slot = next(s for s in rep.slots if s.slot == "X")
        self.assertTrue(slot.actual_censored)
        self.assertIsNotNone(slot.hit)
        self.assertIn("下界", slot.note)

    def test_uncensored_eval_week_uses_interval_hit(self):
        requests, events, sessions = self._data()
        # 用 W1/W2 训练、评估也落在未删失的低需求周：手动切到 as_of=199
        rep = run_backtest(requests, events, sessions, as_of=199,
                           eval_weeks=["W2"])
        slot = next(s for s in rep.slots if s.slot == "X")
        self.assertFalse(slot.actual_censored)
        self.assertIsNotNone(slot.hit)


if __name__ == "__main__":
    unittest.main()
