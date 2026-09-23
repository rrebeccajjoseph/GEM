#!/bin/sh
# Downloads the public rasters used for the compatibility terms.
# Köppen-Geiger: Beck et al. 2018 (figshare). GHSL population: JRC.
set -e

mkdir -p data/rasters/koppen_geiger data/rasters/pop_density

# Note: the figshare.com/ndownloader host answers curl with an empty
# HTTP 202; ndownloader.figshare.com serves the file directly.
curl -fL --retry 5 --retry-all-errors -o data/rasters/koppen_geiger/Beck_KG_V1.zip \
    https://ndownloader.figshare.com/files/12407516
unzip -o data/rasters/koppen_geiger/Beck_KG_V1.zip -d data/rasters/koppen_geiger

curl -L -o data/rasters/pop_density/ghsl.zip \
    https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2022A/GHS_POP_E2020_GLOBE_R2022A_54009_1000/V1-0/GHS_POP_E2020_GLOBE_R2022A_54009_1000_V1_0.zip
unzip -o data/rasters/pop_density/ghsl.zip -d data/rasters/pop_density

# Natural Earth 10m (public domain): admin-0/admin-1 polygons and land, for
# the map rasters built by `python -m energy.maps` (country, region,
# drive_side, coast_km).
mkdir -p data/rasters/natural_earth
for f in ne_10m_admin_0_countries ne_10m_admin_1_states_provinces ne_10m_land; do
    curl -fL --retry 5 -o data/rasters/natural_earth/$f.geojson \
        https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/$f.geojson
done

# WorldClim v2 annual tavg/prec and a global elevation GeoTIFF (e.g.
# GMTED2010) are licensed for manual download; pass their paths to
# `python -m energy.grid` via --worldclim-tavg / --worldclim-prec / --elevation.
echo "Done."
