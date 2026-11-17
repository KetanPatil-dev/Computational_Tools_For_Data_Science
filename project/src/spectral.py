"""Spectral clustering of the product graph.

Nodes are products; edges are the frequent pairs found by A-Priori/PCY (both give the
same pairs), weighted by lift. The graph can be weighted: a node's degree is the sum of
its edge weights.

Main functions:
  product_graph, keep_top_neighbours   build and sparsify the graph
  laplacian, smallest_eigenpairs       Laplacian (L or L_sym) and its smallest eigenvectors
  fiedler_split, recursive_bipartition split by the sign of the Fiedler vector (2 or k clusters)
  spectral_clustering                  k eigenvectors + own k-means (Ng, Jordan & Weiss 2002)
  eigengap, modularity, normalised_cut choose k and evaluate the clusters
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh

WEIGHTS = ("binary", "count", "lift", "log_lift")


# Building the graph

def product_graph(rules, weight="log_lift", min_lift=1.0):
    """
        Build the product graph: one edge per frequent pair with lift > min_lift, 
        weighted by the chosen weight; returns W and the product ids
    """
    if weight not in WEIGHTS:
        raise ValueError(f"weight must be one of {WEIGHTS}")
    pairs = rules[(rules["antecedent"] < rules["consequent"]) & (rules["lift"] > min_lift)]
    a = pairs["antecedent"].to_numpy()
    b = pairs["consequent"].to_numpy()
    w = {
        "binary": np.ones(len(pairs)),
        "count": pairs["count"].to_numpy(dtype=float),
        "lift": pairs["lift"].to_numpy(dtype=float),
        "log_lift": np.log(pairs["lift"].to_numpy(dtype=float)),}[weight]
    ids = np.unique(np.concatenate([a, b]))
    i, j = np.searchsorted(ids, a), np.searchsorted(ids, b)
    n = len(ids)
    W = sp.coo_matrix((w, (i, j)), shape=(n, n)).tocsr()
    return (W + W.T).tocsr(), ids


def keep_top_neighbours(W, m):
    """Keep only each product's m strongest edges, removing the weak links 
    of very popular products """

    W = W.tocsr()
    rows, cols, vals = [], [], []
    for i in range(W.shape[0]):
        s, e = W.indptr[i], W.indptr[i + 1]
        best = np.argsort(-W.data[s:e], kind="stable")[:m]
        rows.append(np.full(len(best), i))
        cols.append(W.indices[s:e][best])
        vals.append(W.data[s:e][best])
    K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=W.shape)
    return K.maximum(K.T).tocsr()


def components(W):
    """Connected-component label of every node, with label 0 for the largest component"""
    _, labels = connected_components(W, directed=False)
    order = np.argsort(-np.bincount(labels), kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return rank[labels]


def subgraph(W, ids, keep):
    """The subgraph on the nodes where the boolean mask keep is True"""
    return W[keep][:, keep].tocsr(), ids[keep]


# Laplacian and its eigenvectors

def degrees(W):
    return np.asarray(W.sum(axis=1)).ravel()


def laplacian(W, kind="sym"):
    """Laplacian of the graph: unnormalised L = D - W, or normalised L_sym = I - D^-1/2 W D^-1/2"""

    d = degrees(W)
    if kind == "unnormalized":
        return (sp.diags(d) - W).tocsr()
    if kind == "sym":
        s = sp.diags(1 / np.sqrt(d))
        return (sp.identity(len(d)) - s @ W @ s).tocsr()
    raise ValueError("kind must be 'unnormalized' or 'sym'")


def smallest_eigenpairs(W, k, kind="sym", solver="sparse", seed=0):
    """The k smallest eigenvalues of the Laplacian and their eigenvectors, 
    computed with a dense (eigh) or sparse (eigsh) solver"""

    n = W.shape[0]
    if solver == "dense":
        vals, vecs = np.linalg.eigh(laplacian(W, kind).toarray())
        vals, vecs = vals[:k], vecs[:, :k]
    elif solver == "sparse":
        v0 = np.random.default_rng(seed).random(n)
        if kind == "sym":
            s = sp.diags(1 / np.sqrt(degrees(W)))
            mu, vecs = eigsh(s @ W @ s, k=k, which="LA", v0=v0)
            vals = 1 - mu
        else:
            vals, vecs = eigsh(laplacian(W, "unnormalized"), k=k, sigma=-1e-3, which="LM", v0=v0)
        order = np.argsort(vals)
        vals, vecs = vals[order], vecs[:, order]
    else:
        raise ValueError("solver must be 'dense' or 'sparse'")
    vals = np.clip(vals, 0, None)                               # remove tiny negative round-off at lambda_1 = 0
    flip = np.sign(vecs[np.abs(vecs).argmax(axis=0), np.arange(vecs.shape[1])])
    return vals, vecs * flip


def eigengap(vals, k_min=2):
    """Suggest the number of clusters k: the k with the largest gap between lambda_k and lambda_{k+1}"""
    gaps = np.diff(vals)
    return int(np.argmax(gaps[k_min - 1:]) + k_min), gaps


# k-means

def _sq_dist(X, C):
    return np.maximum((X ** 2).sum(1)[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None, :], 0)


def _kmeans_pp(X, k, rng):
    """k-means++: each next centre is drawn with probability proportional to its squared distance"""

    centres = [X[rng.integers(len(X))]]
    d = _sq_dist(X, centres[0][None])[:, 0]
    for _ in range(1, k):
        p = d / d.sum() if d.sum() > 0 else np.full(len(X), 1 / len(X))
        centres.append(X[rng.choice(len(X), p=p)])
        d = np.minimum(d, _sq_dist(X, centres[-1][None])[:, 0])
    return np.array(centres)


def kmeans(X, k, n_init=10, max_iter=300, seed=0):
    """Lloyd's k-means with k-means++ starts; the best of n_init runs (lowest inertia)"""

    rng = np.random.default_rng(seed)
    best_labels, best_inertia = None, np.inf
    for _ in range(n_init):
        C = _kmeans_pp(X, k, rng)
        labels = np.full(len(X), -1)
        for _ in range(max_iter):
            new = _sq_dist(X, C).argmin(1)
            if np.array_equal(new, labels):
                break
            labels = new
            for c in range(k):
                members = X[labels == c]
                # an empty cluster restarts at the point farthest from its centre
                C[c] = members.mean(0) if len(members) else X[_sq_dist(X, C).min(1).argmax()]
        inertia = _sq_dist(X, C)[np.arange(len(X)), labels].sum()
        if inertia < best_inertia:
            best_labels, best_inertia = labels, inertia
    return best_labels, best_inertia


# Spectral clustering

def spectral_embedding(W, k, kind="sym", solver="sparse", seed=0):
    """Turn each node into a point in R^k using the k smallest eigenvectors"""

    vals, vecs = smallest_eigenpairs(W, k, kind, solver, seed)
    if kind == "sym":
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    return vals, vecs


def spectral_clustering(W, k, kind="sym", solver="sparse", n_init=10, seed=0):
    """k clusters: k-means on the spectral embedding. Returns cluster labels 0..k-1"""

    _, X = spectral_embedding(W, k, kind, solver, seed)
    labels, _ = kmeans(X, k, n_init=n_init, seed=seed)
    return relabel_by_size(labels)


def fiedler_split(W, kind="unnormalized", solver="dense", split="sign"):
    """Split the graph in two using the Fiedler vector x2: by its sign (x_i < 0 -> X), 
    or by the best threshold (split="kmeans")"""

    vals, vecs = smallest_eigenpairs(W, 2, kind, solver)
    x2 = vecs[:, 1]
    if kind == "sym":
        x2 = x2 / np.sqrt(degrees(W))
    if split == "sign":
        return x2 < 0, vals[1], x2
    if split != "kmeans":
        raise ValueError("split must be 'sign' or 'kmeans'")
    s = np.sort(x2)
    n = len(s)
    left_n = np.arange(1, n)
    left_sum, left_sq = np.cumsum(s)[:-1], np.cumsum(s ** 2)[:-1]
    right_sum, right_sq = s.sum() - left_sum, (s ** 2).sum() - left_sq
    sse = left_sq - left_sum ** 2 / left_n + right_sq - right_sum ** 2 / (n - left_n)
    t = int(np.argmin(sse))
    return x2 <= s[t], vals[1], x2


def recursive_bipartition(W, k, kind="sym", split="sign", dense_below=1000):
    """k clusters by repeatedly splitting the largest cluster in two with 
    fiedler_split (lecture's "idea 1")"""

    labels = np.zeros(W.shape[0], dtype=int)
    for new in range(1, k):
        c = np.bincount(labels).argmax()
        idx = np.flatnonzero(labels == c)
        S = W[idx][:, idx]
        comp = components(S)
        if comp.max() > 0:
            side = comp != 0
        else:
            side, _, _ = fiedler_split(S, kind, "dense" if len(idx) < dense_below else "sparse", split)
        labels[idx[side]] = new
    return relabel_by_size(labels)


def relabel_by_size(labels):
    """Rename clusters 0, 1, 2, ... from largest to smallest (ties by first appearance)"""

    uniq, inv, counts = np.unique(labels, return_inverse=True, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return rank[inv]


# Evaluation

def modularity(W, labels):
    """Weighted modularity of a clustering (same value as networkx's modularity)"""

    labels = np.asarray(labels)
    two_m = W.sum()
    vol = np.bincount(labels, weights=degrees(W))
    Wc = W.tocoo()
    same = labels[Wc.row] == labels[Wc.col]
    w_in = np.bincount(labels[Wc.row[same]], weights=Wc.data[same], minlength=len(vol))
    return float((w_in / two_m - (vol / two_m) ** 2).sum())


def normalised_cut(W, labels):
    """Ncut = sum_c cut(c, rest) / vol(c), the k-way version of the lecture's ncut(A, B)"""

    labels = np.asarray(labels)
    vol = np.bincount(labels, weights=degrees(W))
    Wc = W.tocoo()
    cross = labels[Wc.row] != labels[Wc.col]
    cut = np.bincount(labels[Wc.row[cross]], weights=Wc.data[cross], minlength=len(vol))
    return float((cut / vol).sum())


def cluster_table(labels, ids, products, W=None, top=5):
    """Summary table with one row per cluster: size, main department and aisle, and its top products"""

    df = pd.DataFrame({"cluster": labels, "product": info["product_name"].to_numpy(),
                       "aisle": info["aisle"].to_numpy(), "department": info["department"].to_numpy(),
                       "degree": degrees(W) if W is not None else 0.0})
    rows = []
    for c, g in df.groupby("cluster"):
        dep, ais = g["department"].value_counts(), g["aisle"].value_counts()
        rows.append({"cluster": c, "size": len(g),
                    "top department": f"{dep.index[0]} ({dep.iloc[0] / len(g):.0%})",
                    "top aisle": f"{ais.index[0]} ({ais.iloc[0] / len(g):.0%})",
                    "examples": "; ".join(g.nlargest(top, "degree")["product"]),})
    return pd.DataFrame(rows)
