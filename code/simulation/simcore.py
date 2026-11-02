"""Core of the synthetic simulation: topic banks, passage generation, persistence features.

Self-contained implementation (numpy, scipy, scikit-learn, ripser). Conventions:

* Topic bank ("qr"): the N columns of the orthonormal factor Q of a D x N standard-normal matrix
  (numpy default_rng(seed)), each divided by its norm. Requires N <= D.
* Topic bank ("random", superposition regime): N independent standard-normal vectors in R^D, normalised;
  for N > D they are not orthogonal (|cos| ~ 1/sqrt(D)).
* A passage with t topics drawn from a pool: t distinct topic ids sampled without replacement from the pool
  (generator rng), the sum of their unit vectors, plus isotropic Gaussian noise of scale sigma per coordinate
  (drawn from numpy's *global* legacy RNG, which the drivers re-seed), then L2-normalised.
* Persistence features of a query q and its retrieved set C (rows), under the eps rule:
    keep neighbour i if ||c_i - q|| / min_j ||c_j - q|| - 1 <= eps (eps = inf keeps all), need >= 3 kept;
    offsets u_i = (c_i - q + machine_eps) / ||.||; Vietoris-Rips persistence (ripser, Z/2, maxdim 1);
    W1(H0)       = mean over finite H0 deaths d of d/2;
    LT_max(H1)   = max over H1 points (b, d) of (d - b)/2  (NaN here when there is no H1 point; the
                   analyses that use all 50 neighbours replace NaN by 0, the paper's convention);
    longest bar  = max finite H0 death (H0 births are 0).
"""
import numpy as np
from ripser import ripser
from sklearn.metrics.pairwise import euclidean_distances

MACHINE_EPS = np.finfo(float).eps


def topic_bank_qr(n_topics, dim, seed):
    """Orthonormal topic directions, one per row (n_topics x dim)."""
    G = np.random.default_rng(seed).standard_normal((dim, n_topics))
    Q, _ = np.linalg.qr(G)
    rows = []
    for i in range(n_topics):
        v = Q[:, i]
        rows.append(v / np.linalg.norm(v))
    return np.array(rows)


def topic_bank_random(n_topics, dim, seed):
    """Independent random unit directions, one per row (superposition regime when n_topics > dim)."""
    M = np.random.default_rng(seed).standard_normal((n_topics, dim))
    M /= np.linalg.norm(M, axis=1, keepdims=True)
    return np.array([M[i] for i in range(n_topics)])


class PassageGenerator:
    """Draws passages as noisy normalised sums of topic vectors."""

    def __init__(self, topics, seed):
        self.topics = topics                      # (N, D)
        self.n_topics, self.dim = topics.shape
        self.rng = np.random.default_rng(seed)    # topic-subset draws

    def sample(self, n_topics, pool=None, n_points=1, sigma=0.0):
        """Return n_points passages, each a dict with X (unit vector), n_features (= n_topics) and
        features_ids (the topic ids used). pool = topic ids to draw from (default: a fresh random subset of
        size n_topics of all topics). Noise comes from the global numpy RNG, one D-vector per passage."""
        if pool is None:
            pool = self.rng.choice(np.arange(0, self.n_topics, 1), size=n_topics, replace=False)
        ids = np.vstack([self.rng.choice(pool, size=n_topics, replace=False) for _ in range(n_points)])
        X = self.topics[ids].sum(axis=1)
        if sigma > 0:
            for i in range(n_points):
                X[i] += np.random.normal(scale=sigma, size=self.dim)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        return [dict(X=X[i], n_features=n_topics, features_ids=ids[i]) for i in range(n_points)]


def _diagram_features(U):
    dg = ripser(U, maxdim=1)["dgms"]
    h0 = dg[0][:, 1]
    h0 = h0[np.isfinite(h0)]
    lt = dg[1][:, 1] - dg[1][:, 0]
    return (float(np.mean(h0 / 2.0)),
            float(np.max(lt) / 2.0) if len(lt) else np.nan,
            float(np.max(h0)))


def persistence_features(q, C, eps=np.inf):
    """Features of query q (D,) against retrieved set C (n, D) under the eps rule (see module docstring).
    Returns dict(w1_h0, ltmax_h1, h0_maxbar, nneighbours); the three features are NaN if < 3 neighbours kept."""
    dist = euclidean_distances(np.vstack([q[None, :], C]))[1:, 0]
    order = np.argsort(dist)
    d_sorted, C_sorted = dist[order], C[order]
    keep = d_sorted / d_sorted.min() - 1 <= eps
    n_in = int(keep.sum())
    out = dict(w1_h0=np.nan, ltmax_h1=np.nan, h0_maxbar=np.nan, nneighbours=n_in)
    if n_in < 3:
        return out
    U = C_sorted[keep] - q[None, :] + MACHINE_EPS
    U = U / np.linalg.norm(U, axis=-1)[:, None]
    out["w1_h0"], out["ltmax_h1"], out["h0_maxbar"] = _diagram_features(U)
    return out


def matched_cosine_null(q, C, rng):
    """Matched-cosine null: c_i' = a_i q + sqrt(1 - a_i^2) v_i with a_i = cos(c_i, q) and v_i a fresh uniform
    unit vector orthogonal to q. Returns (unit q, null cloud)."""
    qn = q / np.linalg.norm(q)
    Cn = C / np.linalg.norm(C, axis=1, keepdims=True)
    a = np.clip(Cn @ qn, -1.0, 1.0)
    G = rng.standard_normal(Cn.shape)
    G -= np.outer(G @ qn, qn)
    G /= np.linalg.norm(G, axis=1, keepdims=True)
    return qn, a[:, None] * qn[None, :] + np.sqrt(1.0 - a ** 2)[:, None] * G
