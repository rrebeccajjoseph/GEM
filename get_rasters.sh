#!/bin/sh
# Downloads the public rasters used for the compatibility terms.
# Köppen-Geiger: Beck et al. 2018 (figshare). GHSL population: JRC.
set -e

mkdir -p data/rasters/koppen_geiger data/rasters/pop_density

curl -L -o data/rasters/koppen_geiger/Beck_KG_V1.zip \
    https://figshare.com/ndownloader/files/12407516
unzip -o data/rasters/koppen_geiger/Beck_KG_V1.zip -d data/rasters/koppen_geiger

curl -L -o data/rasters/pop_density/ghsl.zip \
    https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2022A/GHS_POP_E2020_GLOBE_R2022A_54009_1000/V1-0/GHS_POP_E2020_GLOBE_R2022A_54009_1000_V1_0.zip
unzip -o data/rasters/pop_density/ghsl.zip -d data/rasters/pop_density

# WorldClim v2 annual tavg/prec and a global elevation GeoTIFF (e.g.
# GMTED2010) are licensed for manual download; pass their paths to
# `python -m energy.grid` via --worldclim-tavg / --worldclim-prec / --elevation.
echo "Done."
