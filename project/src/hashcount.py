"""Sparse counting for pairs/triples (the "triples method" of MMDS 6.2.1).

Only itemsets that actually occur get a slot, in an open-addressing hash table
with int64 keys and int32 counts that doubles when half full. Used where the
candidates are too sparse for a triangular matrix (PCY's last pass, triples).
"""
import numpy as np
from numba import njit

EMPTY = -1
TABLE_MULT = 0x5851F42D4C957F2D


@njit(cache=True)
def mix(key, mult):
    """Multiplicative hash of an int64 key; result is non-negative."""
    x = key * mult
    x ^= x >> 31
    return x & 0x7FFFFFFFFFFFFFFF


@njit(cache=True)
def new_table(capacity):
    return np.full(capacity, EMPTY, np.int64), np.zeros(capacity, np.int32)


@njit(cache=True)
def _insert(keys, vals, key, add):
    mask = len(keys) - 1
    h = mix(key, TABLE_MULT) & mask
    while True:
        k = keys[h]
        if k == key:
            vals[h] += add
            return 0
        if k == EMPTY:
            keys[h] = key
            vals[h] = add
            return 1
        h = (h + 1) & mask


@njit(cache=True)
def increment(keys, vals, used, key):
    """Count key once more. Returns the table (grown if it got half full) and its new size."""
    used += _insert(keys, vals, key, 1)
    if used * 2 > len(keys):
        new_keys, new_vals = new_table(len(keys) * 2)
        for p in range(len(keys)):
            if keys[p] != EMPTY:
                _insert(new_keys, new_vals, keys[p], vals[p])
        return new_keys, new_vals, used
    return keys, vals, used


def table_items(keys, vals, min_count):
    """(keys, counts) of all entries with count >= min_count."""
    hit = (keys != EMPTY) & (vals >= min_count)
    return keys[hit], vals[hit]
