"""潜在需求估计：删失泊松 + 经验贝叶斯收缩，附不确定区间与证据分级。

核心判断
--------
- 满额（售罄/零名额）是右删失观测：只看到了前 d 个到达者，真实需求 D>=d，
  不能把 d 当需求，更不能把"没留下请求的满额时段"当零需求。
- 闭馆不是删失，是"无观测"，不进入似然，只在证据质量里说明。
- 每个 slot-week 是一个暴露单位；用 EM 不动点求泊松到达率 λ 的极大似然，
  再用 Gamma 共轭先验做经验贝叶斯收缩，得到点估计与可信区间。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .reconstruct import Reconstruction, SlotWeekObs

GRADE_A = "A"          # 资料较足：多次未删失观测
GRADE_B = "B"          # 有一定历史支持
GRADE_C = "C"          # 不能从历史支持：无观测/仅匿名/新品类或节庆
NEW_CONDITION_TAGS = {"节庆", "新品类", "新讲师", "新场地"}


@dataclass
class _Obs:
    week: str
    d: int                  # 有身份的观测需求（rank1 独立人数）
    censored: bool
    threshold: int          # 删失阈值 c=max(d, 名额)；未删失为 0
    anonymous_views: int
    zero_capacity: bool
    capacity_unknown: bool


@dataclass
class SlotDemandEstimate:
    slot: str
    point: float | None                 # 每周独立（rank1）申请者期望人数
    ci_low: float | None
    ci_high: float | None
    observed_mean: float | None         # 仅未删失周的均值
    weeks_total: int
    uncensored_weeks: int
    censored_weeks: int
    closed_weeks: int
    anonymous_full_views: int
    waitlist_persons: int
    zero_capacity_weeks: int
    capacity_unknown_weeks: int
    grade: str
    grade_reasons: list[str] = field(default_factory=list)
    posterior_alpha: float | None = None
    posterior_beta: float | None = None

    def as_dict(self) -> dict:
        return {
            "slot": self.slot,
            "expected_weekly_demand": _round(self.point),
            "ci90": [_round(self.ci_low), _round(self.ci_high)],
            "observed_uncensored_mean": _round(self.observed_mean),
            "weeks": {
                "total": self.weeks_total,
                "uncensored": self.uncensored_weeks,
                "censored": self.censored_weeks,
                "closed": self.closed_weeks,
            },
            "anonymous_full_views": self.anonymous_full_views,
            "waitlist_persons": self.waitlist_persons,
            "grade": self.grade,
            "grade_reasons": self.grade_reasons,
        }


def _round(x: float | None, nd: int = 3) -> float | None:
    return None if x is None else round(x, nd)


# ---- 泊松工具 ---------------------------------------------------------------

def _poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * lam**k / math.factorial(k)


def _poisson_cdf(k: int, lam: float) -> float:
    if k < 0:
        return 0.0
    return math.fsum(_poisson_pmf(i, lam) for i in range(k + 1))


def _censored_effective(lam: float, c: int) -> float:
    """E[D | D >= c]，泊松删失观测在 EM 中的填补值。

    恒等式 E[D; D>=c] = λ P(D >= c-1)，故
    E[D|D>=c] = λ P(D>=c-1) / P(D>=c)。
    c=0 时条件恒真，返回 λ（该观测对似然无信息）。
    """
    if c <= 0:
        return lam
    denom = 1.0 - _poisson_cdf(c - 1, lam)
    if denom <= 0.0:
        return float(c)  # 数值兜底
    numer = 1.0 - _poisson_cdf(c - 2, lam)
    return lam * numer / denom


def _fit_pooled_lambda(obs_list: list[_Obs], tol: float = 1e-7,
                       max_iter: int = 500) -> float:
    informative = [o for o in obs_list if not (o.censored and o.threshold == 0)]
    if not informative:
        return float("nan")
    lam = max(
        0.5,
        sum(o.d for o in obs_list if not o.censored)
        / max(1, sum(1 for o in obs_list if not o.censored)),
    )
    for _ in range(max_iter):
        total = 0.0
        for o in informative:
            if not o.censored:
                total += o.d
            else:
                total += _censored_effective(lam, o.threshold)
        new_lam = total / len(informative)
        if abs(new_lam - lam) <= tol * max(1.0, lam):
            return new_lam
        lam = new_lam
    return lam


# ---- Gamma 分位数 -----------------------------------------------------------

def _gamma_quantile(p: float, shape: float, rate: float) -> float:
    """Gamma(shape, rate) 的 p 分位数；优先不完全伽马函数二分，否则 WH 近似。"""
    if hasattr(math, "gammainc"):
        target = p

        def cdf(x: float) -> float:
            return math.gammainc(shape, rate * x)

        lo, hi = 1e-9, 1.0
        while cdf(hi) < p:
            hi *= 2.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if cdf(mid) < target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    # Wilson–Hilferty 卡方近似（shape<1 时退化为均值附近的保守区间）
    if shape >= 1.0:
        z = 1.281551566 if p >= 0.5 else -1.281551566  # 90% 区间 ±
        # 由调用方只在 0.05/0.95 使用；这里通用正态分位近似
        z = _norm_quantile(p)
        chi = 2 * shape * (1 - 1 / (9 * shape) + z / math.sqrt(9 * shape)) ** 3
        return max(0.0, chi / (2 * rate))
    mean = shape / rate
    return mean * (0.5 if p < 0.5 else 1.5)


def _norm_quantile(p: float) -> float:
    # Beasley–Springer–Moro 风格的逆正态，足够估计区间用
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
           ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


# ---- 主流程 -----------------------------------------------------------------

def _slot_obs(ob: SlotWeekObs) -> _Obs | None:
    if ob.closed:
        return None  # 闭馆：无观测
    d = ob.observed_demand
    if ob.censored:
        q = ob.sellout_capacity
        if q is None:
            q = ob.min_capacity if ob.min_capacity is not None else d
        return _Obs(week=ob.week, d=d, censored=True,
                    threshold=max(d, q),
                    anonymous_views=ob.anonymous_full_views,
                    zero_capacity=ob.zero_capacity,
                    capacity_unknown=ob.capacity_unknown)
    return _Obs(week=ob.week, d=d, censored=False, threshold=0,
                anonymous_views=ob.anonymous_full_views,
                zero_capacity=False, capacity_unknown=ob.capacity_unknown)


def _fit_gamma_prior(slot_means: list[float], pooled: float) -> tuple[float, float]:
    """矩估计 Gamma 先验；样本不足时退化为以 pooled 为中心的弱先验。"""
    if len(slot_means) >= 3:
        mean = sum(slot_means) / len(slot_means)
        var = sum((x - mean) ** 2 for x in slot_means) / (len(slot_means) - 1)
        if var > 1e-9 and mean > 0:
            shape = mean**2 / var
            rate = mean / var
            if shape > 0 and rate > 0:
                return shape, rate
    center = pooled if not math.isnan(pooled) and pooled > 0 else 1.0
    return 2.0, 2.0 / center  # 弱先验：α=2，均值=center


def estimate_demand(
    recon: Reconstruction,
    plan_tags: dict[str, list[str]] | None = None,
) -> dict[str, SlotDemandEstimate]:
    plan_tags = plan_tags or {}
    slots = sorted({o.slot for o in recon.observations.values()})

    per_slot: dict[str, list[_Obs]] = {}
    raw: dict[str, SlotWeekObs]
    for slot in slots:
        obs_list = []
        for ob in recon.slot_weeks(slot):
            parsed = _slot_obs(ob)
            if parsed is not None:
                obs_list.append(parsed)
        per_slot[slot] = obs_list

    all_obs = [o for lst in per_slot.values() for o in lst]
    pooled = _fit_pooled_lambda(all_obs)

    uncensored_means = [
        sum(o.d for o in lst) / len([o for o in lst if not o.censored])
        for lst in per_slot.values()
        if any(not o.censored for o in lst)
    ]
    alpha0, beta0 = _fit_gamma_prior(uncensored_means, pooled)

    out: dict[str, SlotDemandEstimate] = {}
    for slot in slots:
        obs_list = per_slot[slot]
        weeks = recon.slot_weeks(slot)
        closed_n = sum(1 for o in weeks if o.closed)
        anon = sum(o.anonymous_full_views for o in weeks)
        wait_n = sum(len(o.final_waitlist) for o in weeks)
        zero_n = sum(1 for o in weeks if o.zero_capacity)
        cap_unk_n = sum(1 for o in weeks if o.capacity_unknown)
        uncens = [o for o in obs_list if not o.censored]
        cens = [o for o in obs_list if o.censored]
        reasons: list[str] = []

        if not obs_list:
            out[slot] = SlotDemandEstimate(
                slot=slot, point=None, ci_low=None, ci_high=None,
                observed_mean=None, weeks_total=len(weeks),
                uncensored_weeks=0, censored_weeks=0, closed_weeks=closed_n,
                anonymous_full_views=anon, waitlist_persons=wait_n,
                zero_capacity_weeks=zero_n, capacity_unknown_weeks=cap_unk_n,
                grade=GRADE_C,
                grade_reasons=["历史上没有任何可用于该时段的需求观测（或全部闭馆）"],
            )
            continue

        uncens_mean = (sum(o.d for o in uncens) / len(uncens)) if uncens else None

        # EM 插补删失周，得到后验（Gamma-泊松共轭，删失用插补值的近似）
        lam_local = _fit_pooled_lambda(obs_list)
        if math.isnan(lam_local):
            lam_local = pooled if not math.isnan(pooled) else 1.0
        eff_sum = 0.0
        for o in obs_list:
            eff_sum += o.d if not o.censored else _censored_effective(
                lam_local, o.threshold)
        alpha_post = alpha0 + eff_sum
        beta_post = beta0 + len(obs_list)
        point = alpha_post / beta_post
        ci_low = _gamma_quantile(0.05, alpha_post, beta_post)
        ci_high = _gamma_quantile(0.95, alpha_post, beta_post)

        # 证据分级：删失周只给下界，A 级要求足够多的"未删失"直接观测
        if len(uncens) >= 3:
            grade = GRADE_A
            reasons.append(f"有 {len(uncens)} 个未删失周的直接观测")
        elif len(uncens) >= 1:
            grade = GRADE_B
            reasons.append(
                f"仅 {len(uncens)} 个未删失周、{len(cens)} 个删失周，"
                "真实需求主要由删失模型外推，区间较宽")
        else:
            grade = GRADE_C
            reasons.append("没有任何未删失周：所有观测都被满额/零名额截断，"
                           "无法直接验证需求水平")
        if cens:
            reasons.append(f"{len(cens)} 个满额/零名额周为右删失，真实需求高于观测值")
        if zero_n:
            reasons.append(f"{zero_n} 个名额设零周：删失观测（区别于闭馆的无观测）")
        if closed_n:
            reasons.append(f"{closed_n} 个闭馆周不参与估计，属无观测而非零需求")
        if anon:
            reasons.append(f"{anon} 次匿名满额浏览：存在未留下身份的需求，无法计入人数")
        if cap_unk_n:
            reasons.append(f"{cap_unk_n} 个周名额未知，按未删失弱证据处理")
        tag_hits = [t for t in plan_tags.get(slot, []) if t in NEW_CONDITION_TAGS]
        if tag_hits:
            grade = GRADE_C
            reasons.append(f"新品类/节庆条件 {tag_hits}：历史不可直接外推，建议先小范围试开")

        out[slot] = SlotDemandEstimate(
            slot=slot, point=point, ci_low=ci_low, ci_high=ci_high,
            observed_mean=uncens_mean, weeks_total=len(weeks),
            uncensored_weeks=len(uncens), censored_weeks=len(cens),
            closed_weeks=closed_n, anonymous_full_views=anon,
            waitlist_persons=wait_n, zero_capacity_weeks=zero_n,
            capacity_unknown_weeks=cap_unk_n, grade=grade,
            grade_reasons=reasons, posterior_alpha=alpha_post,
            posterior_beta=beta_post,
        )
    return out
