"""Own A-Priori for frequent pairs (MMDS ch. 6).

Pass 1 counts single items; pass 2 counts only pairs of frequent items, stored
with the triangular-matrix method so every pair has a fixed slot in one array.
"""
import numpy as np
import pandas as pd
from numba import njit

from .hashcount import increment, new_table, table_items

# Triangular matrix for n frequent items uses n(n-1)/2 int32 counters; refuse beyond this.
MAX_TRIANGLE_BYTES = 2 * 1024**3


def tri_starts(n):
    """Slot of pair (i, i+1) for each i, i.e. where row i of the triangle begins."""
    i = np.arange(n, dtype=np.int64)
    return i * (2 * n - i - 1) // 2


@njit(cache=True)
def _count_pairs_triangular(indptr, items, index_of, n, max_basket):
    counts = np.zeros(n * (n - 1) // 2, dtype=np.int32)
    buf = np.empty(max_basket, dtype=np.int64)
    for b in range(len(indptr) - 1):
        m = 0
        for p in range(indptr[b], indptr[b + 1]):
            idx = index_of[items[p]]
            if idx >= 0:
                buf[m] = idx
                m += 1
        if m < 2:
            continue
        s = np.sort(buf[:m])
        for x in range(m - 1):
            i = s[x]
            if x > 0 and i == s[x - 1]:
                continue
            base = i * (2 * n - i - 1) // 2 - i - 1
            for y in range(x + 1, m):
                if s[y] == s[y - 1]:
                    continue
                counts[base + s[y]] += 1
    return counts


def apriori_pairs(baskets, min_count):
    """Frequent pairs with support count >= min_count.

    Returns (pairs, item_counts): pairs is a DataFrame with item_a < item_b and
    count; item_counts[p] is the number of baskets containing product p.
    """
    item_counts = np.bincount(baskets.items)
    frequent = np.flatnonzero(item_counts >= min_count)
    n = len(frequent)
    if n * (n - 1) // 2 * 4 > MAX_TRIANGLE_BYTES:
        raise MemoryError(
            f"{n} frequent items need {n * (n - 1) // 2 * 4 / 1e9:.1f} GB of pair counters; "
            "raise min_count (or use PCY to prune pairs first)."
        )
    index_of = np.full(len(item_counts), -1, dtype=np.int64)
    index_of[frequent] = np.arange(n)

    counts = _count_pairs_triangular(
        baskets.indptr, baskets.items, index_of, n, int(baskets.sizes().max())
    )
    k = np.flatnonzero(counts >= min_count)
    starts = tri_starts(n)
    i = np.searchsorted(starts, k, side="right") - 1
    j = k - starts[i] + i + 1
    pairs = pd.DataFrame({"item_a": frequent[i], "item_b": frequent[j], "count": counts[k]})
    return pairs.sort_values("count", ascending=False, ignore_index=True), item_counts


@njit(cache=True)
def _count_triples(indptr, items, index_of, pair_ok, max_basket):
    n = pair_ok.shape[0]
    keys, vals = new_table(1 << 16)
    used = 0
    buf = np.empty(max_basket, dtype=np.int64)
    for b in range(len(indptr) - 1):
        m = 0
        for p in range(indptr[b], indptr[b + 1]):
            idx = index_of[items[p]]
            if idx >= 0:
                buf[m] = idx
                m += 1
        if m < 3:
            continue
        s = np.unique(buf[:m])
        m = len(s)
        for x in range(m - 2):
            for y in range(x + 1, m - 1):
                if not pair_ok[s[x], s[y]]:
                    continue
                for z in range(y + 1, m):
                    if pair_ok[s[x], s[z]] and pair_ok[s[y], s[z]]:
                        keys, vals, used = increment(keys, vals, used, (s[x] * n + s[y]) * n + s[z])
    return keys, vals, used


def apriori_triples(baskets, pairs, min_count):
    """Frequent triples: A-Priori's third pass, counting only triples whose three pairs are all frequent."""
    members = np.unique(np.concatenate([pairs["item_a"], pairs["item_b"]]))
    n = len(members)
    index_of = np.full(int(baskets.items.max()) + 1, -1, dtype=np.int64)
    index_of[members] = np.arange(n)
    pair_ok = np.zeros((n, n), dtype=np.bool_)
    a, b = index_of[pairs["item_a"]], index_of[pairs["item_b"]]
    pair_ok[a, b] = pair_ok[b, a] = True

    keys, vals, n_candidates = _count_triples(
        baskets.indptr, baskets.items, index_of, pair_ok, int(baskets.sizes().max())
    )
    keys, vals = table_items(keys, vals, min_count)
    triples = pd.DataFrame({
        "item_a": members[keys // (n * n)],
        "item_b": members[keys // n % n],
        "item_c": members[keys % n],
        "count": vals,
    })
    return triples.sort_values("count", ascending=False, ignore_index=True), n_candidates


def association_rules(pairs, item_counts, n_baskets):
    """Both rules a -> b and b -> a for every frequent pair, with support, confidence and lift."""
    a = np.concatenate([pairs["item_a"], pairs["item_b"]])
    b = np.concatenate([pairs["item_b"], pairs["item_a"]])
    c = np.concatenate([pairs["count"], pairs["count"]]).astype(np.float64)
    rules = pd.DataFrame({
        "antecedent": a,
        "consequent": b,
        "count": c.astype(np.int64),
        "support": c / n_baskets,
        "confidence": c / item_counts[a],
        "lift": c * n_baskets / (item_counts[a].astype(np.float64) * item_counts[b]),
    })
    return rules.sort_values("lift", ascending=False, ignore_index=True)
