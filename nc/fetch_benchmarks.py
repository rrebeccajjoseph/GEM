"""Fetch the Im2GPS3k / YFCC4k images that are still publicly reachable.

The published archives (MediaFire mirrors) are gone. What remains:
  - Im2GPS3k file names are Flickr's id_secret_server_owner.jpg, so the
    original is https://live.staticflickr.com/{server}/{id}_{secret}.jpg
  - YFCC4k rows carry the YFCC100M static URL (rebuilt on live.staticflickr)
    and the YFCC100M file hash, which keys the AWS Multimedia Commons mirror —
    it still holds some photos their owners since deleted from Flickr.

About 82% of each came back in spot checks. Deleted photos are gone for
good, so this yields SUBSETS: comparable across our own checkpoints (all
scored on the same images), not with published numbers. The complete
copies are on Delta.

Writes data/benchmarks/<name>_images/<IMG_ID>, and registers <name> in
benchmarks.json with a meta CSV filtered to the images that arrived
(<name>_avail.csv); the full metadata stays registered as <name>_full for
the overlap id pass, which needs no images.

Usage: python nc/fetch_benchmarks.py [--workers 16]
"""
import os, re, csv, json, time, argparse, urllib.request
from concurrent.futures import ThreadPoolExecutor

STATIC = re.compile(r'staticflickr\.com/(\d+)/(\d+)_([0-9a-f]+)\.jpg')
HEADERS = {'User-Agent': 'spherical-pigeon-benchmark-fetch/1.0 (research)'}


def fetch(url: str, dest: str) -> bool:
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as r:
            if 'photo_unavailable' in r.geturl():
                return False
            data = r.read()
        if len(data) < 5000:        # Flickr's "unavailable" placeholder is tiny
            return False
        with open(dest + '.part', 'wb') as fh:
            fh.write(data)
        os.replace(dest + '.part', dest)
        return True
    except Exception:
        return False


def candidates(name: str, row: list, header: list) -> tuple:
    """(image file name, candidate URLs in order)."""
    if name == 'im2gps3k':
        img = row[header.index('IMG_ID')]
        pid, secret, server = img.split('_')[:3]
        return img, [f'https://live.staticflickr.com/{server}/{pid}_{secret}.jpg']
    img = row[header.index('IMG_ID')]
    urls = []
    m = STATIC.search(','.join(row))
    if m:
        urls.append(f'https://live.staticflickr.com/{m.group(1)}/{m.group(2)}_{m.group(3)}.jpg')
    h = row[2].zfill(32)   # YFCC100M file hash; CSV round-trips dropped leading zeros
    if re.fullmatch(r'[0-9a-f]{32}', h):
        urls.append('https://multimedia-commons.s3-us-west-2.amazonaws.com/data/images/'
                    f'{h[:3]}/{h[3:6]}/{h}.jpg')
    return img, urls


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument('--workers', type=int, default=16)
    argp.add_argument('--dir', default='data/benchmarks')
    args = argp.parse_args()

    path = os.path.join(args.dir, 'benchmarks.json')
    reg = json.load(open(path)) if os.path.exists(path) else {}
    for name in ('im2gps3k', 'yfcc4k'):
        meta = os.path.join(args.dir, f'{name}_places365.csv')
        out = os.path.join(args.dir, f'{name}_images')
        os.makedirs(out, exist_ok=True)
        with open(meta) as fh:
            rows = list(csv.reader(fh))
        header, body = rows[0], rows[1:]

        def one(row):
            img, urls = candidates(name, row, header)
            dest = os.path.join(out, img)
            if os.path.exists(dest):
                return row, True
            for u in urls:
                if fetch(u, dest):
                    return row, True
                time.sleep(0.2)
            return row, False

        t0 = time.time()
        with ThreadPoolExecutor(args.workers) as ex:
            results = list(ex.map(one, body))
        got = [r for r, ok in results if ok]
        avail = os.path.join(args.dir, f'{name}_avail.csv')
        with open(avail, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(got)
        reg[name] = {'meta': avail, 'images': out}
        reg[f'{name}_full'] = {'meta': meta, 'images': out}
        print(f'{name}: {len(got)}/{len(body)} images ({len(got) / len(body):.1%}) '
              f'in {time.time() - t0:.0f}s -> {avail}')
    json.dump(reg, open(path, 'w'), indent=2)


if __name__ == '__main__':
    main()
