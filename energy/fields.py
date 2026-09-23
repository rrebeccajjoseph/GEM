"""Raster fields: raw raster values on an equirectangular lat/lng array,
sampled at arbitrary points on-device.

The grid table (energy.grid.build_table) answers "what is the raster over
cell g"; a field answers "what is the raster at (lat, lng)" — the lookup a
cell-free model needs, since its quadrature points and inference iterates
are not cell centroids. Fields hold RAW values (NaN = no data); RasterBank
applies its own normalization (log1p, standardization, class casting) on
top, so a field and the table go through the same transform.

Pixel convention: array [H, W] with H = 180 / res_deg, W = 360 / res_deg;
row 0 is the band just below lat 90, column 0 the band just east of lng
-180. Pixel (i, j) is centered at (90 - (i + 0.5) res, -180 + (j + 0.5) res).
Longitude wraps; latitude clamps at the poles.

Built by `python -m energy.grid --fields-out data/energy/fields.npz`.
"""

import torch
import numpy as np
from torch import nn, Tensor


class RasterFields(nn.Module):
    """Every field in a fields npz, as device buffers.

    Buffers are non-persistent: fields are data, rebuilt from the npz, not
    model state — they never enter a checkpoint.

    Args:
        arrays (dict): name -> np.ndarray [H, W] raw values, NaN = no data
        res_deg (float): pixel size in degrees (same for every field)
    """

    def __init__(self, arrays: dict, res_deg: float):
        super().__init__()
        self.res_deg = float(res_deg)
        self.names = sorted(arrays.keys())
        for name in self.names:
            arr = np.asarray(arrays[name], dtype=np.float32)
            expected = (round(180 / self.res_deg), round(360 / self.res_deg))
            assert arr.shape == expected, \
                f'field {name}: shape {arr.shape}, expected {expected} at {res_deg} deg'
            self.register_buffer(f'field_{name}', torch.from_numpy(arr), persistent=False)

    @classmethod
    def load(cls, path: str) -> 'RasterFields':
        data = np.load(path, allow_pickle=False)
        arrays = {k[len('field_'):]: data[k] for k in data.files if k.startswith('field_')}
        return cls(arrays, float(data['res_deg']))

    def __contains__(self, name: str) -> bool:
        return name in self.names

    def sample(self, name: str, latlng: Tensor, mode: str='bilinear') -> Tensor:
        """Raw values of one field at points.

        Args:
            name (str): field name
            latlng (Tensor): degrees [N, 2] (lat, lng); any lng, lat clamped
            mode (str): 'nearest' (categorical) or 'bilinear' (continuous).
                Bilinear renormalizes over the valid corners, so a point next
                to a coastline takes the land value instead of NaN-poisoning;
                it is NaN only when all four corners are.

        Returns:
            Tensor: [N] float32, NaN where there is no data
        """
        field = getattr(self, f'field_{name}')
        H, W = field.shape
        lat, lng = latlng[:, 0].float(), latlng[:, 1].float()
        # continuous pixel coordinates: integer values land on pixel centers
        i_f = (90.0 - lat) / self.res_deg - 0.5
        j_f = (lng + 180.0) / self.res_deg - 0.5

        if mode == 'nearest':
            i = torch.round(i_f).long().clamp(0, H - 1)
            j = torch.remainder(torch.round(j_f).long(), W)
            return field[i, j]
        if mode != 'bilinear':
            raise ValueError(f'Unknown sampling mode: {mode}')

        i0 = torch.floor(i_f)
        j0 = torch.floor(j_f)
        di, dj = i_f - i0, j_f - j0
        i0 = i0.long()
        j0 = j0.long()
        rows = (i0.clamp(0, H - 1), (i0 + 1).clamp(0, H - 1))
        cols = (torch.remainder(j0, W), torch.remainder(j0 + 1, W))
        weights = ((1 - di) * (1 - dj), (1 - di) * dj, di * (1 - dj), di * dj)
        corners = ((0, 0), (0, 1), (1, 0), (1, 1))

        num = torch.zeros_like(lat)
        den = torch.zeros_like(lat)
        for w, (r, c) in zip(weights, corners):
            v = field[rows[r], cols[c]]
            ok = ~torch.isnan(v)
            # zero NaNs BEFORE multiplying: torch.where(ok, w * v, 0) would
            # still backpropagate 0 * NaN = NaN into w, and through it into
            # the coordinate being optimized (energy.infer)
            v = torch.where(ok, v, torch.zeros_like(v))
            w = w * ok
            num = num + w * v
            den = den + w
        return torch.where(den > 0, num / den.clamp(min=1e-12),
                           torch.full_like(num, float('nan')))
