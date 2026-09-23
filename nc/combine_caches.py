"""OSV-5M cache rows then MP-16 cache rows, as one cache with one index.
MP-16 ids get an mp16_ prefix (OSV-5M ids are bare numbers too), which is
why the overlap exclusion list needs a prefixed twin for this cache."""
import os
import numpy as np
import pandas as pd

a, b, out = 'data/energy/cache', 'data/energy/cache_mp16', 'data/energy/cache_combined'
os.makedirs(out, exist_ok=True)
ia = pd.read_csv(f'{a}/index.csv', dtype={'id': str})
ib = pd.read_csv(f'{b}/index.csv', dtype={'id': str})
ib['id'] = 'mp16_' + ib['id']
ea = np.load(f'{a}/embeddings.f16.npy', mmap_mode='r')
eb = np.load(f'{b}/embeddings.f16.npy', mmap_mode='r')
assert len(ia) == len(ea) and len(ib) == len(eb), 'index/embedding row mismatch'
e = np.lib.format.open_memmap(f'{out}/embeddings.f16.npy', mode='w+', dtype=np.float16,
                              shape=(len(ea) + len(eb), ea.shape[1]))
step = 1 << 20
for s in range(0, len(ea), step):
    e[s:s + step] = ea[s:s + step]
for s in range(0, len(eb), step):
    e[len(ea) + s:len(ea) + s + step] = eb[s:s + step]
e.flush()
pd.concat([ia, ib], ignore_index=True)[['id', 'lat', 'lng', 'selection']] \
    .to_csv(f'{out}/index.csv', index=False)
print(f'combined {len(ea)} OSV-5M + {len(eb)} MP-16 rows -> {out}')
