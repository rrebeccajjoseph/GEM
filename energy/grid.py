"""H3 grid construction and the raster compatibility lookup table.

Builds the res-4 grid G (288,122 near-equal-area cells) and samples every
raster over each cell into A in [|G|, K]. Values are aggregated over the cell
area, not the centroid: each cell is covered by its res-(r+offset) children,
whose centroids give a quasi-uniform equal-area sample of the hexagon —
modal aggregation for categorical rasters, mean for continuous ones. This
avoids polygon rasterization entirely.

Samplers are injectable callables (lats, lngs) -> values so the table builder
is testable without raster files on disk.

The same rasters are also written as lat/lng fields (--fields-out): each
GeoTIFF reprojected straight onto an equirectangular pixel grid, for
point lookups off the cell grid (energy.fields.RasterFields).

Usage (run from the repository root, after get_auxiliary_data.sh):

    python -m energy.grid --out data/energy/grid.npz [--resolution 4] \
        [--fields-out data/energy/fields.npz [--fields-res 0.1] [--fields-only]]
"""

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
if project_dir not in sys.path:
    sys.path.append(project_dir)

import logging
import argparse
import numpy as np
import pandas as pd
import h3

logger = logging.getLogger('energy.grid')
logging.basicConfig(level=logging.INFO)

# h3 v3/v4 API compatibility
if hasattr(h3, 'get_res0_cells'):
    _res0 = h3.get_res0_cells
    _children = h3.cell_to_children
    _to_latlng = h3.cell_to_latlng
    _to_cell = h3.latlng_to_cell
else:  # h3 v3
    _res0 = h3.get_res0_indexes
    _children = h3.h3_to_children
    _to_latlng = h3.h3_to_geo
    _to_cell = h3.geo_to_h3


def build_grid(resolution: int=4):
    """Builds the full H3 grid at the given resolution.

    Args:
        resolution (int, optional): H3 resolution. Defaults to 4 (288,122 cells).

    Returns:
        tuple: (cells: list of H3 indexes sorted for determinism,
                latlngs: np.ndarray [G, 2] of cell centroids)
    """
    cells = []
    for base in _res0():
        cells.extend(_children(base, resolution))

    cells = sorted(cells)
    latlngs = np.array([_to_latlng(c) for c in cells], dtype=np.float64)
    logger.info(f'Built H3 res-{resolution} grid: {len(cells)} cells.')
    return cells, latlngs


def cell_index_map(cells: list) -> dict:
    """Maps H3 index -> position in the grid arrays."""
    return {c: i for i, c in enumerate(cells)}


def snap_to_grid(lats: np.ndarray, lngs: np.ndarray, resolution: int,
                 index_map: dict) -> np.ndarray:
    """Snaps coordinates to grid positions.

    Args:
        lats (np.ndarray): latitudes
        lngs (np.ndarray): longitudes
        resolution (int): grid H3 resolution
        index_map (dict): from cell_index_map

    Returns:
        np.ndarray: integer grid positions
    """
    return np.array([index_map[_to_cell(lat, lng, resolution)]
                     for lat, lng in zip(lats, lngs)], dtype=np.int64)


def aggregate_labels_over_cells(cells: list, resolution: int, lats: np.ndarray,
                                lngs: np.ndarray, labels: np.ndarray) -> tuple:
    """Builds a per-cell categorical raster from OBSERVED training rows —
    e.g. country/region, already sitting in per-image metadata (OSV-5M's
    `country`/`region`/`sub-region`, MP-16-Pro's `country`/`region`), unlike
    aggregate_over_cells which samples a continuous *external* raster. No
    geocoding needed: every row already carries its own label, this just
    votes those labels into the grid.

    A cell with at least one observed row takes the majority label among
    them. A cell with none (sparse ocean/desert/polar regions, or simply
    outside every dataset pooled in) is filled from the nearest cell that
    does have a majority, by cell-centroid distance — the same nearest-fill
    convention osv5m.py's climate() uses for ocean/nodata Köppen pixels.

    Args:
        cells: H3 grid cells
        resolution: grid resolution
        lats, lngs: observed row coordinates (pool multiple datasets by
            concatenating before calling — one consistent factorization)
        labels: observed row labels (any hashable, e.g. country strings)

    Returns:
        tuple: (raster [G] float64, int-coded with NaN nowhere after fill;
                categories: list where categories[code] recovers the label)
    """
    from scipy.spatial import cKDTree

    idx_map = cell_index_map(cells)
    pos = snap_to_grid(lats, lngs, resolution, idx_map)
    codes, categories = pd.factorize(labels)

    G = len(cells)
    votes = [{} for _ in range(G)]
    for p, c in zip(pos, codes):
        if c < 0:  # factorize's code for a missing/NaN label
            continue
        votes[p][c] = votes[p].get(c, 0) + 1

    raster = np.full(G, np.nan)
    has_vote = np.zeros(G, dtype=bool)
    for g, v in enumerate(votes):
        if v:
            raster[g] = max(v, key=v.get)
            has_vote[g] = True

    n_empty = int((~has_vote).sum())
    if n_empty:
        latlngs = np.array([_to_latlng(c) for c in cells])
        tree = cKDTree(latlngs[has_vote])
        _, nearest = tree.query(latlngs[~has_vote], k=1)
        raster[~has_vote] = raster[has_vote][nearest]
        logger.info(f'{n_empty}/{G} cells had no observed row; filled from '
                    f'the nearest cell that did.')

    return raster, list(categories)


def aggregate_values_over_cells(cells: list, resolution: int, lats: np.ndarray,
                                lngs: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Continuous counterpart to aggregate_labels_over_cells: mean of the
    OBSERVED rows' values per cell (e.g. OSV-5M's dist_sea, MP-16's
    Prob_indoor/natural/urban), not an external raster sample. Same
    nearest-fill for cells with no observed row.

    Returns:
        np.ndarray: raster [G] float64, no NaN after fill.
    """
    from scipy.spatial import cKDTree

    idx_map = cell_index_map(cells)
    pos = snap_to_grid(lats, lngs, resolution, idx_map)
    values = np.asarray(values, dtype=np.float64)

    G = len(cells)
    sums = np.zeros(G)
    counts = np.zeros(G)
    ok = ~np.isnan(values)
    np.add.at(sums, pos[ok], values[ok])
    np.add.at(counts, pos[ok], 1)

    has_val = counts > 0
    raster = np.full(G, np.nan)
    raster[has_val] = sums[has_val] / counts[has_val]

    n_empty = int((~has_val).sum())
    if n_empty:
        latlngs = np.array([_to_latlng(c) for c in cells])
        tree = cKDTree(latlngs[has_val])
        _, nearest = tree.query(latlngs[~has_val], k=1)
        raster[~has_val] = raster[has_val][nearest]
        logger.info(f'{n_empty}/{G} cells had no observed row; filled from '
                    f'the nearest cell that did.')

    return raster


def aggregate_over_cells(cells: list, sampler, agg: str, resolution: int,
                         child_offset: int=2, batch_size: int=2048) -> np.ndarray:
    """Samples a raster at child centroids of each cell and aggregates.

    Children of an H3 cell at resolution r+offset are near-equal-area, so
    their centroid samples approximate an area-weighted aggregate.

    Args:
        cells (list): H3 cells of the grid
        sampler (callable): (lats [N], lngs [N]) -> values [N]; NaN = nodata
        agg (str): 'modal' (categorical) or 'mean' (continuous)
        resolution (int): grid resolution
        child_offset (int, optional): child resolution offset (2 -> 49 pts/cell)
        batch_size (int, optional): cells per sampler call batch

    Returns:
        np.ndarray: aggregated values [G], NaN where every sample was nodata
    """
    child_res = resolution + child_offset
    out = np.full(len(cells), np.nan)

    for start in range(0, len(cells), batch_size):
        batch = cells[start:start + batch_size]
        child_lists = [list(_children(c, child_res)) for c in batch]
        counts = [len(ch) for ch in child_lists]
        flat = [c for ch in child_lists for c in ch]
        pts = np.array([_to_latlng(c) for c in flat], dtype=np.float64)
        vals = np.asarray(sampler(pts[:, 0], pts[:, 1]), dtype=np.float64)

        pos = 0
        for i, n in enumerate(counts):
            cell_vals = vals[pos:pos + n]
            pos += n
            cell_vals = cell_vals[~np.isnan(cell_vals)]
            if len(cell_vals) == 0:
                continue

            if agg == 'modal':
                uniq, cnt = np.unique(cell_vals, return_counts=True)
                out[start + i] = uniq[np.argmax(cnt)]
            elif agg == 'mean':
                out[start + i] = cell_vals.mean()
            else:
                raise ValueError(f'Unknown aggregation: {agg}')

    return out


class RasterioSampler:
    """Point sampler over a GeoTIFF via rasterio (EPSG:4326 rasters).

    Args:
        path (str): raster file path
        band (int, optional): band to read. Defaults to 1.
        nodata (float, optional): explicit nodata value -> NaN.
        transform_value (callable, optional): applied to valid values.
    """

    def __init__(self, path: str, band: int=1, nodata: float=None,
                 transform_value=None):
        import rasterio
        self.dataset = rasterio.open(path)
        self.band = band
        self.nodata = nodata if nodata is not None else self.dataset.nodata
        self.transform_value = transform_value
        # Reproject query points when the raster is not in EPSG:4326 —
        # GHSL is Mollweide (ESRI:54009); sampling it with raw geographic
        # coordinates silently returns nodata everywhere.
        self.needs_reproject = (self.dataset.crs is not None
                                and self.dataset.crs.to_string() != 'EPSG:4326')

    def __call__(self, lats: np.ndarray, lngs: np.ndarray) -> np.ndarray:
        if self.needs_reproject:
            from rasterio.warp import transform as warp_transform
            xs, ys = warp_transform('EPSG:4326', self.dataset.crs,
                                    list(lngs), list(lats))
            coords = list(zip(xs, ys))
        else:
            coords = list(zip(lngs, lats))  # rasterio expects (x, y)
        vals = np.array([v[self.band - 1] for v in self.dataset.sample(coords)],
                        dtype=np.float64)
        if self.nodata is not None:
            vals[vals == self.nodata] = np.nan

        if self.transform_value is not None:
            valid = ~np.isnan(vals)
            vals[valid] = self.transform_value(vals[valid])

        return vals


def build_field(sampler: RasterioSampler, agg: str, res_deg: float) -> np.ndarray:
    """Reprojects a raster onto the equirectangular field grid (see
    energy.fields for the pixel convention), aggregating over each output
    pixel: GDAL mode resampling for categorical rasters, average for
    continuous ones — the per-pixel analogue of aggregate_over_cells.

    Args:
        sampler (RasterioSampler): the source raster (its nodata and
            transform_value apply exactly as for point sampling)
        agg (str): 'modal' or 'mean'
        res_deg (float): output pixel size in degrees

    Returns:
        np.ndarray: [180/res, 360/res] float32, NaN = no data
    """
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_origin

    resampling = {'modal': Resampling.mode, 'mean': Resampling.average}[agg]
    H, W = round(180 / res_deg), round(360 / res_deg)
    out = np.full((H, W), np.nan, dtype=np.float32)
    reproject(source=rasterio.band(sampler.dataset, sampler.band), destination=out,
              src_nodata=sampler.nodata, dst_transform=from_origin(-180, 90, res_deg, res_deg),
              dst_crs='EPSG:4326', dst_nodata=np.nan, resampling=resampling)

    if sampler.transform_value is not None:
        valid = ~np.isnan(out)
        out[valid] = sampler.transform_value(out[valid])
    return out


def build_fields(specs: dict, res_deg: float) -> dict:
    """Builds every raster in specs as a field: name -> [H, W] float32."""
    fields = {}
    for name, (factory, agg) in specs.items():
        logger.info(f'Reprojecting raster to a {res_deg} deg field: {name} ({agg}).')
        fields[name] = build_field(factory(), agg, res_deg)
        logger.info(f'{name}: {np.isnan(fields[name]).mean():.1%} of pixels with no data.')
    return fields


def compare_fields_to_table(fields: dict, res_deg: float, latlngs: np.ndarray,
                            table: dict) -> None:
    """Logs how well the fields reproduce the grid table at cell centroids.

    They are not expected to match exactly — the table aggregates over the
    whole cell, a field over one pixel — but a categorical agreement far
    below ~90% or a low correlation means the two builds disagree about the
    raster (wrong nodata, CRS, or value shift), not just about resolution.
    """
    import torch
    from energy.fields import RasterFields

    rf = RasterFields(fields, res_deg)
    pts = torch.from_numpy(np.asarray(latlngs, dtype=np.float32))
    for name in sorted(set(fields) & set(table)):
        mode = 'nearest' if name == 'climate' else 'bilinear'
        at = rf.sample(name, pts, mode).numpy().astype(np.float64)
        ref = np.asarray(table[name], dtype=np.float64)
        both = ~np.isnan(at) & ~np.isnan(ref)
        coverage = both.sum() / max((~np.isnan(ref)).sum(), 1)
        if mode == 'nearest':
            score = f'class agreement {(at[both] == ref[both]).mean():.1%}'
        else:
            score = f'correlation {np.corrcoef(at[both], ref[both])[0, 1]:.3f}'
        logger.info(f'field vs table, {name}: {score} over {both.sum()} cells '
                    f'(field has data at {coverage:.1%} of the table\'s valid cells).')


def default_raster_specs():
    """The raster table schema: name -> (sampler factory, aggregation).

    Built lazily so the module imports without raster files present.
    Köppen values are shifted to 0-29 (matching climate_zone in the metadata
    CSV); 0 in the raw raster is ocean/nodata.
    """
    from config import KOPPEN_GEIGER_PATH, GHSL_PATH

    def koppen():
        return RasterioSampler(KOPPEN_GEIGER_PATH, nodata=0,
                               transform_value=lambda v: v - 1)

    def popdens():
        return RasterioSampler(GHSL_PATH)

    return {
        'climate': (koppen, 'modal'),
        'popdens': (popdens, 'mean'),
        # temp / precip (WorldClim) and elevation are registered by the CLI
        # via --worldclim-tavg / --worldclim-prec / --elevation once those
        # rasters are downloaded; see main().
    }


def build_table(cells: list, resolution: int, specs: dict) -> dict:
    """Builds the raster lookup table.

    Args:
        cells (list): H3 grid cells
        resolution (int): grid resolution
        specs (dict): name -> (sampler_factory, agg)

    Returns:
        dict: name -> np.ndarray [G]
    """
    table = {}
    for name, (factory, agg) in specs.items():
        logger.info(f'Sampling raster: {name} ({agg}).')
        table[name] = aggregate_over_cells(cells, factory(), agg, resolution)
        n_missing = np.isnan(table[name]).sum()
        logger.info(f'{name}: {n_missing}/{len(cells)} cells with no data '
                    f'(ocean expected for land rasters).')
    return table


def main():
    argp = argparse.ArgumentParser(description='Build H3 grid + raster table.')
    argp.add_argument('--out', default='data/energy/grid.npz')
    argp.add_argument('--resolution', type=int, default=4)
    argp.add_argument('--worldclim-tavg', default=None,
                      help='Path to WorldClim annual mean temperature GeoTIFF.')
    argp.add_argument('--worldclim-prec', default=None,
                      help='Path to WorldClim annual precipitation GeoTIFF.')
    argp.add_argument('--elevation', default=None,
                      help='Path to a global elevation GeoTIFF (e.g. GMTED2010).')
    argp.add_argument('--fields-out', default=None,
                      help='Also write every raster as a lat/lng field (energy.fields) '
                           'to this npz, for point lookups off the cell grid.')
    argp.add_argument('--fields-res', type=float, default=0.1,
                      help='Field pixel size in degrees (0.1 = ~11 km, 1800x3600 '
                           'float32 = 26 MB per raster).')
    argp.add_argument('--fields-only', action='store_true', default=False,
                      help='Build only --fields-out, not the grid table (which '
                           'must already exist at --out for the agreement check).')
    args = argp.parse_args()

    cells, latlngs = build_grid(args.resolution)

    specs = default_raster_specs()
    if args.worldclim_tavg:
        specs['temp'] = (lambda: RasterioSampler(args.worldclim_tavg), 'mean')
    if args.worldclim_prec:
        specs['precip'] = (lambda: RasterioSampler(args.worldclim_prec), 'mean')
    if args.elevation:
        specs['elevation'] = (lambda: RasterioSampler(args.elevation), 'mean')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.fields_only:
        data = np.load(args.out, allow_pickle=False)
        table = {k[len('raster_'):]: data[k] for k in data.files if k.startswith('raster_')}
    else:
        table = build_table(cells, args.resolution, specs)
        np.savez_compressed(
            args.out,
            cells=np.array([str(c) for c in cells]),
            latlngs=latlngs,
            resolution=args.resolution,
            **{f'raster_{k}': v for k, v in table.items()},
        )
        logger.info(f'Saved grid + raster table to {args.out}.')

    if args.fields_out:
        fields = build_fields(specs, args.fields_res)
        compare_fields_to_table(fields, args.fields_res, latlngs, table)
        os.makedirs(os.path.dirname(args.fields_out), exist_ok=True)
        np.savez_compressed(args.fields_out, res_deg=args.fields_res,
                            **{f'field_{k}': v for k, v in fields.items()})
        logger.info(f'Saved {len(fields)} fields to {args.fields_out}.')


if __name__ == '__main__':
    main()
