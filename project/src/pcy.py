"""PCY and its two refinements, Multistage and Multihash (MMDS 6.3).

Pass 1 also hashes every pair to a bucket. A pair can only be frequent if its
bucket is, so the last pass counts just the pairs of frequent items that land
in frequent buckets, in a sparse hash table instead of A-Priori's triangle.

  pcy        one hash table in pass 1
  multistage an extra pass rehashes surviving pairs with a second, independent hash
  multihash  two half-size hash tables in pass 1
"""
import time

import numpy as np
import pandas as pd
from numba import njit

from .hashcount import increment, mix, new_table, table_items

KEY_BASE = 1 << 20  # product ids < 2^20, so key = a * KEY_BASE + b identifies the pair
BUCKET_MULTS = np.array([0x2545F4914F6CDD1D, 0x1B873593A1B2C3D5], dtype=np.int64)
METHODS = ("pcy", "multistage", "multihash")


@njit(cache=True)
def _passes_filters(key, bitmaps, filter_mults):
    for f in range(len(filter_mults)):
        if not bitmaps[f, mix(key, filter_mults[f]) % bitmaps.shape[1]]:
            return False
    return True


@njit(cache=True)
def _hash_pairs(indptr, items, item_ok, n_buckets, count_mults, bitmaps, filter_mults, max_basket):
    """Bucket counts per hash in count_mults, over pairs of ok items that pass every earlier bitmap."""
    counts = np.zeros((len(count_mults), n_buckets), np.int32)
    buf = np.empty(max_basket, np.int64)
    for b in range(len(indptr) - 1):
        m = 0
        for p in range(indptr[b], indptr[b + 1]):
            if item_ok[items[p]]:
                buf[m] = items[p]
                m += 1
        if m < 2:
            continue
        s = np.sort(buf[:m])
        for x in range(m - 1):
            if x > 0 and s[x] == s[x - 1]:
                continue
            for y in range(x + 1, m):
                if s[y] == s[y - 1]:
                    continue
                key = s[x] * KEY_BASE + s[y]
                if _passes_filters(key, bitmaps, filter_mults):
                    for c in range(len(count_mults)):
                        counts[c, mix(key, count_mults[c]) % n_buckets] += 1
    return counts


@njit(cache=True)
def _count_candidates(indptr, items, item_ok, bitmaps, filter_mults, max_basket):
    """Exact counts of the candidate pairs: both items frequent, every bitmap says frequent bucket."""
    keys, vals = new_table(1 << 16)
    used = 0
    buf = np.empty(max_basket, np.int64)
    for b in range(len(indptr) - 1):
        m = 0
        for p in range(indptr[b], indptr[b + 1]):
            if item_ok[items[p]]:
                buf[m] = items[p]
                m += 1
        if m < 2:
            continue
        s = np.sort(buf[:m])
        for x in range(m - 1):
            if x > 0 and s[x] == s[x - 1]:
                continue
            for y in range(x + 1, m):
                if s[y] == s[y - 1]:
                    continue
                key = s[x] * KEY_BASE + s[y]
                if _passes_filters(key, bitmaps, filter_mults):
                    keys, vals, used = increment(keys, vals, used, key)
    return keys, vals, used


def frequent_pairs(baskets, min_count, method="pcy", n_buckets=1 << 22):
    """Frequent pairs by PCY / Multistage / Multihash.

    Returns (pairs, stats). pairs has the same format as apriori.apriori_pairs;
    stats records memory and candidate counts for the comparison with A-Priori.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    t0 = time.perf_counter()
    indptr, items = baskets.indptr, baskets.items
    max_basket = int(baskets.sizes().max())
    item_counts = np.bincount(items)
    item_ok = item_counts >= min_count
    all_items = np.ones_like(item_ok)
    no_bitmaps, no_mults = np.zeros((0, 1), np.bool_), np.zeros(0, np.int64)

    if method == "pcy":
        counts = _hash_pairs(indptr, items, all_items, n_buckets, BUCKET_MULTS[:1], no_bitmaps, no_mults, max_basket)
        bitmaps = counts >= min_count
    elif method == "multistage":
        first = _hash_pairs(indptr, items, all_items, n_buckets, BUCKET_MULTS[:1], no_bitmaps, no_mults, max_basket)
        first = first >= min_count
        counts = _hash_pairs(indptr, items, item_ok, n_buckets, BUCKET_MULTS[1:2], first, BUCKET_MULTS[:1], max_basket)
        bitmaps = np.vstack([first, counts >= min_count])
    else:
        counts = _hash_pairs(indptr, items, all_items, n_buckets // 2, BUCKET_MULTS[:2], no_bitmaps, no_mults, max_basket)
        bitmaps = counts >= min_count
    filter_mults = BUCKET_MULTS[: len(bitmaps)]

    keys, vals, n_candidates = _count_candidates(indptr, items, item_ok, bitmaps, filter_mults, max_basket)
    keys, vals = table_items(keys, vals, min_count)
    pairs = pd.DataFrame({"item_a": keys // KEY_BASE, "item_b": keys % KEY_BASE, "count": vals})
    pairs = pairs.sort_values(["count", "item_a", "item_b"], ascending=[False, True, True], ignore_index=True)

    n_freq = int(item_ok.sum())
    stats = {
        "method": method,
        "frequent items": n_freq,
        # Share of frequent buckets in each hash table (multistage: 2nd table counts only pass-1 survivors)
        "frequent buckets": " / ".join(f"{f:.1%}" for f in bitmaps.mean(axis=1)),
        "candidate pairs": int(n_candidates),
        "frequent pairs": len(pairs),
        "pass-1 bucket MB": counts.nbytes / 1e6,
        # Last pass: bitmaps (1 bit/bucket) + 12 bytes per candidate pair (triples method)
        "last-pass MB": (bitmaps.size / 8 + 12 * n_candidates) / 1e6,
        "seconds": time.perf_counter() - t0,
    }
    return pairs, stats


def apriori_memory_mb(n_frequent_items):
    """Last-pass memory of A-Priori's triangular matrix (4 bytes per pair of frequent items)."""
    return 4 * n_frequent_items * (n_frequent_items - 1) / 2 / 1e6
