"""Spatially held-out validation split, shared by osv5m.py and mp16.py.

A random sample of train rows is a leaky val set: OSV-5M images come in
street-view sequences and MP-16 in photo bursts, so 99% of randomly drawn
val images sat within 1 km of a training image (median 108 m), and val
median_km came out 5-8x better than on OSV-5M's own test split, whose images
are all > 1 km from train. Checkpoint selection and the pilot go / no-go both
read val, so both rewarded memorizing places.

Here val is drawn from whole lat/lng blocks (~block_deg, ~11 km at 0.1°),
at most per_block rows from each so val still covers many places. Every
other train row in a chosen block, and every train row within buffer_km of
a val row (a block edge), becomes 'val_buffer': used for neither training
nor validation, so no val image has a training image within buffer_km.
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

EARTH_RADIUS_KM = 6371.0088


def _unit(lat: np.ndarray, lng: np.ndarray) -> np.ndarray:
    la, ln = np.radians(lat), np.radians(lng)
    return np.stack([np.cos(la) * np.cos(ln), np.cos(la) * np.sin(ln), np.sin(la)], axis=1)


def spatial_val_split(df: pd.DataFrame, val_size: int, seed: int=330,
                      block_deg: float=0.1, per_block: int=20,
                      buffer_km: float=1.0) -> pd.Series:
    """Relabels part of the 'train' rows as 'val' and 'val_buffer'.

    Args:
        df (pd.DataFrame): lat, lng and selection columns; only rows with
            selection == 'train' are touched (a test split stays as it is)
        val_size (int): val rows wanted (fewer if train is smaller)
        seed (int): RNG seed
        block_deg (float): block edge in degrees
        per_block (int): most val rows drawn from one block
        buffer_km (float): minimum distance from any val row to any
            remaining train row

    Returns:
        pd.Series: the new selection column, aligned with df.index
    """
    selection = df['selection'].copy()
    train = df.index[selection == 'train']
    if len(train) == 0 or val_size <= 0:
        return selection
    rng = np.random.default_rng(seed)
    lat = df.loc[train, 'lat'].to_numpy()
    lng = df.loc[train, 'lng'].to_numpy()
    block = pd.Series(list(zip(np.floor(lat / block_deg).astype(np.int64),
                               np.floor(lng / block_deg).astype(np.int64))))
    block_id = pd.factorize(block)[0]

    # blocks in random order; within a block, rows in random order. Val is
    # the first val_size rows (by block order) whose within-block rank is
    # below per_block.
    block_order = rng.permutation(block_id.max() + 1)[block_id]
    jitter = rng.random(len(train))
    order = np.lexsort((jitter, block_order))
    rank = np.empty(len(train), dtype=np.int64)
    sorted_blocks = block_id[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_blocks)) + 1]
    run = np.diff(np.r_[starts, len(order)])
    rank[order] = np.arange(len(order)) - np.repeat(starts, run)
    eligible = order[rank[order] < per_block]
    val_pos = eligible[:min(val_size, len(eligible))]

    is_val = np.zeros(len(train), dtype=bool)
    is_val[val_pos] = True
    in_val_block = np.isin(block_id, np.unique(block_id[val_pos]))
    buffer = in_val_block & ~is_val

    # block edges: train rows within buffer_km of any val row
    xyz = _unit(lat, lng)
    chord = 2 * np.sin(buffer_km / EARTH_RADIUS_KM / 2)
    near = cKDTree(xyz[is_val]).query_ball_point(xyz[~is_val & ~buffer], r=chord,
                                                 return_length=True)
    rest = np.flatnonzero(~is_val & ~buffer)
    buffer[rest[near > 0]] = True

    selection.loc[train[is_val]] = 'val'
    selection.loc[train[buffer]] = 'val_buffer'
    return selection
