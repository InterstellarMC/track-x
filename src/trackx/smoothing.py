"""Extended Rauch–Tung–Striebel backward pass for already assigned barrel paths."""

import numpy as np

from .core import wrap
from .precision import initialize, innovation, propagate, update


def smooth_group(b, seeds, tail):
    """Filter and smooth equal-depth paths. Return estimates at the seed exit."""
    x, p, r = initialize(b, seeds)
    first = x.copy()
    history = []
    for hits in tail.T:
        new_r = b.r[hits]
        y, c, f, ok = propagate(x, p, r, new_r)
        if not ok.all():
            raise ValueError("assigned measurement unreachable in replay")
        _, e, l, n = innovation(
            y, c, np.column_stack((b.phi[hits], b.xyz[hits, 2])), new_r, 1.0
        )
        history.append((x, p, y, c, f))
        x, p = update(y, c, e, l, n)
        r = new_r
    for old, cov, pred, prior, jac in reversed(history):
        chol = np.linalg.cholesky(prior)
        gain = np.linalg.solve(
            chol.transpose(0, 2, 1), np.linalg.solve(chol, jac @ cov)
        ).transpose(0, 2, 1)
        delta = x - pred
        delta[:, 0] = wrap(delta[:, 0])
        x = old + (gain @ delta[:, :, None])[:, :, 0]
        x[:, 0] = wrap(x[:, 0])
        p = cov + gain @ (p - prior) @ gain.transpose(0, 2, 1)
        p = (p + p.transpose(0, 2, 1)) / 2
    return first, x, p
