"""Idle-spell survival p-values.

The question for every in-stock SKU is: *how surprising is the current silence?*

The SKU's closed inter-pick gaps (measured in days the SKU was actually in
stock, so stock-outs are not mistaken for neglect) describe how long this
product normally stays quiet.  The current idle spell is a right-censored gap.
We ask: under the SKU's own history, what is the probability of a gap at least
this long?

Two models are fitted and **both must agree** that the spell is unusual:

* an exponential (Poisson-demand) model, which is robust when there are few
  picks, with weak pseudo-counts so a never-picked SKU still gets a sensible rate;
* a log-normal gap model, which respects regular, clockwork demand and
  shrinks its spread toward a typical value when there are few gaps.

The returned p-value is the larger of the two, i.e. the conservative one.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr


def idle_pvalue(
    idle_days,
    n_picks,
    hist_days,
    n_gaps,
    log_gap_mean,
    log_gap_var,
    prior_picks: float = 0.5,
    prior_days: float = 30.0,
    sigma_prior: float = 0.8,
    sigma_weight: float = 4.0,
):
    """Vectorised p-value that an idle spell is at least this long.

    Parameters are per-SKU arrays.  ``idle_days`` is the number of in-stock days
    since the last pick; ``hist_days`` is the in-stock history before that spell.

    Returns ``(p, p_exponential, p_lognormal)``.  Small ``p`` means the silence
    is unusual for this SKU.
    """
    h = np.maximum(np.asarray(idle_days, dtype=float) + 1.0, 1.0)
    n_picks = np.asarray(n_picks, dtype=float)
    hist = np.maximum(np.asarray(hist_days, dtype=float), 0.0)
    n = np.asarray(n_gaps, dtype=float)

    lam = (n_picks + prior_picks) / (hist + prior_days)
    p_exp = np.exp(-lam * h)

    mu = np.nan_to_num(np.asarray(log_gap_mean, dtype=float))
    var_obs = np.nan_to_num(np.asarray(log_gap_var, dtype=float))
    var = (n * var_obs + sigma_weight * sigma_prior**2) / (n + sigma_weight)
    z = (np.log(h) - mu) / np.sqrt(var)
    p_ln = 1.0 - ndtr(z)

    p = np.where(n >= 3, np.maximum(p_exp, p_ln), p_exp)
    return np.clip(p, 1e-12, 1.0), p_exp, p_ln
