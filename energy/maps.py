"""Map rasters: geographic clue terms from real maps, replacing the ones
voted into the grid from training labels (nc/build_geo_rasters.py).

The label-derived rasters were per-cell label statistics: the majority
`country` / `scene` / `road_index` of the training photos that fell in each
res-4 cell, nearest-filled elsewhere. That is a learned-from-the-targets
prior dressed up as a map. It only exists on cells, and for scene /
prob_* / road_index it describes the photos rather than the place. Every
term here is a property of the location itself, defined at any lat/lng:

  country     Natural Earth admin-0 polygons (ADM0_A3), categorical
  region      Natural Earth admin-1 polygons (adm1_code), categorical
  drive_side  1 = left-hand traffic, 0 = right, from the country polygons
              and LEFT_HAND_TRAFFIC below
  coast_km    geodesic distance to the nearest coastline pixel, km (land)
  land_cover  optional GeoTIFF (e.g. ESA WorldCover / CCI-LC), categorical
  soil        optional GeoTIFF (e.g. SoilGrids WRB most-probable), categorical

Each is built once as a lat/lng field (energy.fields). The grid table rows
are then aggregated FROM that field over each cell's children, so the table
and the field share one set of class codes.

Usage (after get_rasters.sh fetched the Natural Earth GeoJSON):
    python -m energy.maps --grid data/energy/grid.npz \
        --out-grid data/energy/grid_maps.npz --fields data/energy/fields.npz \
        [--land-cover path.tif] [--soil path.tif] [--res 0.1]

Then train with --grid data/energy/grid_maps.npz --fields data/energy/fields.npz.
"""

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
if project_dir not in sys.path:
    sys.path.append(project_dir)

import json
import logging
import argparse
import numpy as np

logger = logging.getLogger('energy.maps')
logging.basicConfig(level=logging.INFO)

EARTH_RADIUS_KM = 6371.0

# Countries and territories driving on the left, by ISO 3166-1 alpha-3 /
# Natural Earth ADM0_A3 (source: the "Left- and right-hand traffic" country
# list; stable for decades, the last switch was Samoa in 2009).
LEFT_HAND_TRAFFIC = {
    'AIA', 'ATG', 'AUS', 'BGD', 'BHS', 'BMU', 'BRB', 'BRN', 'BTN', 'BWA', 'CCK',
    'COK', 'CXR', 'CYM', 'CYP', 'DMA', 'FJI', 'FLK', 'GBR', 'GGY', 'GRD', 'GUY',
    'HKG', 'IDN', 'IMN', 'IND', 'IRL', 'JAM', 'JEY', 'JPN', 'KEN', 'KIR', 'KNA',
    'LCA', 'LKA', 'LSO', 'MAC', 'MDV', 'MLT', 'MOZ', 'MSR', 'MUS', 'MWI', 'MYS',
    'NAM', 'NFK', 'NIU', 'NPL', 'NRU', 'NZL', 'PAK', 'PCN', 'PNG', 'SGP', 'SHN',
    'SLB', 'SUR', 'SWZ', 'SYC', 'TCA', 'THA', 'TKL', 'TLS', 'TON', 'TTO', 'TUV',
    'TZA', 'UGA', 'VCT', 'VGB', 'VIR', 'WSM', 'ZAF', 'ZMB', 'ZWE',
}

NE_DIR = 'data/rasters/natural_earth'
NE_FILES = {'admin0': 'ne_10m_admin_0_countries.geojson',
            'admin1': 'ne_10m_admin_1_states_provinces.geojson',
            'land': 'ne_10m_land.geojson'}


def _transform(res_deg: float):
    from rasterio.transform import from_origin
    return from_origin(-180, 90, res_deg, res_deg)


def _shape(res_deg: float) -> tuple:
    return round(180 / res_deg), round(360 / res_deg)


def load_features(path: str) -> list:
    with open(path) as fh:
        return json.load(fh)['features']


def rasterize_categorical(features: list, key, res_deg: float,
                          all_touched: bool=True) -> tuple:
    """Polygons -> an int-coded field (NaN outside every polygon).

    all_touched labels every pixel a polygon touches, not only pixels whose
    centre it contains: at 0.1 deg a harbour-front city's own pixel is often
    centred on water (Sydney's is in Port Jackson), and a coastal photo
    should not lose its country clue to that.

    Args:
        features (list): GeoJSON features
        key (callable): feature -> label (None skips the feature)
        res_deg (float): pixel size
        all_touched (bool): see above; False for a strict land mask

    Returns:
        tuple: (field [H, W] float32, labels list — labels[code] is the label)
    """
    from rasterio.features import rasterize

    labeled = [(f['geometry'], key(f)) for f in features if f.get('geometry')]
    labels = sorted({lab for _, lab in labeled if lab is not None})
    code = {lab: i for i, lab in enumerate(labels)}
    # rasterize burns integers; -1 marks "no polygon" and becomes NaN
    burned = rasterize(((g, code[lab]) for g, lab in labeled if lab is not None),
                       out_shape=_shape(res_deg), transform=_transform(res_deg),
                       fill=-1, dtype='int32', all_touched=all_touched)
    field = burned.astype(np.float32)
    field[burned < 0] = np.nan
    return field, labels


def remap_dense(field: np.ndarray) -> tuple:
    """Sparse class codes (ESA's 10, 20, ..., 220) -> dense 0..K-1, so a
    categorical head is not sized by the largest code. Returns (field,
    original codes list)."""
    valid = ~np.isnan(field)
    codes, dense = np.unique(field[valid], return_inverse=True)
    out = np.full_like(field, np.nan)
    out[valid] = dense
    return out, codes.tolist()


def coast_distance_km(land: np.ndarray, res_deg: float) -> np.ndarray:
    """Geodesic km from every land pixel to the nearest coastline pixel (a
    land pixel with a sea 4-neighbour); NaN over the sea.

    Nearest-neighbour search runs on unit 3-vectors, so the distance is a
    true great-circle distance at every latitude — an equirectangular
    distance transform would inflate east-west distances toward the poles.
    """
    from scipy.spatial import cKDTree

    H, W = land.shape
    sea = ~land
    up = np.roll(sea, 1, axis=0); up[0] = False
    down = np.roll(sea, -1, axis=0); down[-1] = False
    coast = land & (up | down | np.roll(sea, 1, axis=1) | np.roll(sea, -1, axis=1))

    lat = np.deg2rad(90 - (np.arange(H) + 0.5) * res_deg)
    lng = np.deg2rad(-180 + (np.arange(W) + 0.5) * res_deg)

    def unit(ii, jj):
        return np.stack([np.cos(lat[ii]) * np.cos(lng[jj]),
                         np.cos(lat[ii]) * np.sin(lng[jj]), np.sin(lat[ii])], axis=1)

    ci, cj = np.nonzero(coast)
    tree = cKDTree(unit(ci, cj))
    li, lj = np.nonzero(land)
    out = np.full((H, W), np.nan, dtype=np.float32)
    for s in range(0, len(li), 1_000_000):
        chord, _ = tree.query(unit(li[s:s + 1_000_000], lj[s:s + 1_000_000]), k=1)
        out[li[s:s + 1_000_000], lj[s:s + 1_000_000]] = \
            2 * EARTH_RADIUS_KM * np.arcsin(np.clip(chord / 2, 0, 1))
    return out


def build_map_fields(ne_dir: str, res_deg: float, land_cover: str=None,
                     soil: str=None) -> tuple:
    """All map fields at res_deg.

    Returns:
        tuple: (fields dict name -> [H, W] float32, categories dict
                name -> labels list, aggregation dict name -> 'modal'|'mean')
    """
    fields, categories, agg = {}, {}, {}

    admin0 = load_features(os.path.join(ne_dir, NE_FILES['admin0']))
    fields['country'], categories['country'] = rasterize_categorical(
        admin0, lambda f: f['properties'].get('ADM0_A3'), res_deg)
    agg['country'] = 'modal'
    logger.info(f"country: {len(categories['country'])} classes.")

    lht = np.array([lab in LEFT_HAND_TRAFFIC for lab in categories['country']], dtype=np.float32)
    valid = ~np.isnan(fields['country'])
    drive = np.full_like(fields['country'], np.nan)
    drive[valid] = lht[fields['country'][valid].astype(np.int64)]
    fields['drive_side'], agg['drive_side'] = drive, 'modal'
    missing = LEFT_HAND_TRAFFIC - set(categories['country'])
    logger.info(f'drive_side: {int(lht.sum())} left-hand-traffic countries mapped'
                + (f'; not in the polygons (too small at 10m): {sorted(missing)}' if missing else '.'))

    admin1 = load_features(os.path.join(ne_dir, NE_FILES['admin1']))
    fields['region'], categories['region'] = rasterize_categorical(
        admin1, lambda f: f['properties'].get('adm1_code'), res_deg)
    agg['region'] = 'modal'
    logger.info(f"region: {len(categories['region'])} classes.")

    land_features = load_features(os.path.join(ne_dir, NE_FILES['land']))
    # strict centres here: the coastline is where land pixels meet sea pixels
    land_field, _ = rasterize_categorical(land_features, lambda f: 'land', res_deg,
                                          all_touched=False)
    fields['coast_km'] = coast_distance_km(~np.isnan(land_field), res_deg)
    agg['coast_km'] = 'mean'
    logger.info(f"coast_km: max {np.nanmax(fields['coast_km']):.0f} km.")

    from energy.grid import RasterioSampler, build_field
    for name, path in (('land_cover', land_cover), ('soil', soil)):
        if path:
            fields[name], categories[name] = remap_dense(
                build_field(RasterioSampler(path), 'modal', res_deg))
            agg[name] = 'modal'
            logger.info(f'{name}: {len(categories[name])} classes from {path}.')

    return fields, categories, agg


# Rasters that were per-cell training-label statistics (nc/build_geo_rasters.py):
# dropped from any grid this module writes.
LABEL_DERIVED = {'country', 'region', 'subregion', 'land_cover', 'soil', 'road_index',
                 'scene', 'dist_sea', 'prob_indoor', 'prob_natural', 'prob_urban'}


def table_from_fields(fields: dict, agg: dict, res_deg: float, cells: list,
                      resolution: int) -> dict:
    """Grid-table rows aggregated from the fields over each cell's children
    (energy.grid.aggregate_over_cells), so table and field share codes."""
    import torch
    from energy.fields import RasterFields
    from energy.grid import aggregate_over_cells

    rf = RasterFields(fields, res_deg)
    table = {}
    for name in fields:
        mode = 'nearest' if agg[name] == 'modal' else 'bilinear'

        def sampler(lats, lngs, name=name, mode=mode):
            pts = torch.from_numpy(np.stack([lats, lngs], 1).astype(np.float32))
            return rf.sample(name, pts, mode).numpy().astype(np.float64)

        logger.info(f'Aggregating {name} over grid cells ({agg[name]}).')
        table[name] = aggregate_over_cells(cells, sampler, agg[name], resolution)
    return table


def main():
    from energy.grid import build_grid

    argp = argparse.ArgumentParser(description='Build map rasters (fields + grid table).')
    argp.add_argument('--ne-dir', default=NE_DIR)
    argp.add_argument('--grid', default='data/energy/grid.npz',
                      help='Existing grid; its external rasters are kept, label-derived '
                           'ones dropped.')
    argp.add_argument('--out-grid', default='data/energy/grid_maps.npz')
    argp.add_argument('--fields', default='data/energy/fields.npz',
                      help='Fields npz to extend in place (created if missing).')
    argp.add_argument('--res', type=float, default=None,
                      help='Field pixel size (default: the existing fields file\'s, else 0.1).')
    argp.add_argument('--land-cover', default=None)
    argp.add_argument('--soil', default=None)
    args = argp.parse_args()

    existing = {}
    res = args.res or 0.1
    if os.path.exists(args.fields):
        data = np.load(args.fields, allow_pickle=False)
        existing = {k: data[k] for k in data.files}
        file_res = float(existing['res_deg'])
        if args.res and abs(args.res - file_res) > 1e-9:
            raise SystemExit(f'--res {args.res} differs from {args.fields} ({file_res}); '
                             'every field in one file shares a resolution.')
        res = file_res

    fields, categories, agg = build_map_fields(args.ne_dir, res, args.land_cover, args.soil)

    base = dict(np.load(args.grid, allow_pickle=False))
    dropped = sorted(k[len('raster_'):] for k in base
                     if k.startswith('raster_') and k[len('raster_'):] in LABEL_DERIVED)
    base = {k: v for k, v in base.items()
            if not (k.startswith('raster_') and k[len('raster_'):] in LABEL_DERIVED)}
    cells, _ = build_grid(int(base['resolution']))
    table = table_from_fields(fields, agg, res, cells, int(base['resolution']))
    for name, raster in table.items():
        base[f'raster_{name}'] = raster
    # code -> label, outside the raster_ prefix (load_grid reads raster_* as
    # numeric tables with allow_pickle=False); plain unicode, no pickling
    for name, labels in categories.items():
        base[f'categories_{name}'] = np.array([str(x) for x in labels])
    np.savez_compressed(args.out_grid, **base)
    logger.info(f'Wrote {args.out_grid}: map rasters {sorted(table)}; dropped '
                f'label-derived {dropped or "none"}.')

    out = {**existing, 'res_deg': np.float64(res), **{f'field_{k}': v for k, v in fields.items()}}
    np.savez_compressed(args.fields, **out)
    logger.info(f'Wrote {len(fields)} map fields into {args.fields}.')


if __name__ == '__main__':
    main()
