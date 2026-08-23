"""H3 grid construction and the raster compatibility lookup table.

Builds the res-4 grid G (288,122 near-equal-area cells) and samples every
raster over each cell into A in [|G|, K]. Values are aggregated over the cell
area, not the centroid: each cell is covered by its res-(r+offset) children,
whose centroids give a quasi-uniform equal-area sample of the hexagon —
modal aggregation for categorical rasters, mean for continuous ones. This
avoids polygon rasterization entirely.

Samplers are injectable callables (lats, lngs) -> values so the table builder
is testable without raster files on disk.

Usage (run from the repository root, after get_auxiliary_data.sh):

    python -m energy.grid --out data/energy/grid.npz [--resolution 4]
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
    args = argp.parse_args()

    cells, latlngs = build_grid(args.resolution)

    specs = default_raster_specs()
    if args.worldclim_tavg:
        specs['temp'] = (lambda: RasterioSampler(args.worldclim_tavg), 'mean')
    if args.worldclim_prec:
        specs['precip'] = (lambda: RasterioSampler(args.worldclim_prec), 'mean')
    if args.elevation:
        specs['elevation'] = (lambda: RasterioSampler(args.elevation), 'mean')

    table = build_table(cells, args.resolution, specs)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out,
        cells=np.array([str(c) for c in cells]),
        latlngs=latlngs,
        resolution=args.resolution,
        **{f'raster_{k}': v for k, v in table.items()},
    )
    logger.info(f'Saved grid + raster table to {args.out}.')


if __name__ == '__main__':
    main()
