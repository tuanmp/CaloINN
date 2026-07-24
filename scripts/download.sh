#!/bin/bash
# Download FCCeeALLEGRO dataset parts from Zenodo record 17045562
# Usage: bash download.sh
# Resumes partial downloads via curl -C -

set -e

ZENODO="https://zenodo.org/api/records/17045562/files"
FILES=(
  "LEMURS_FCCeeALLEGRO_gamma_100kEvents_1GeV100GeV_GPSflat_part1.h5"
  "LEMURS_FCCeeALLEGRO_gamma_100kEvents_1GeV100GeV_GPSflat_part2.h5"
  "LEMURS_FCCeeALLEGRO_gamma_100kEvents_1GeV100GeV_GPSflat_part3.h5"
  "LEMURS_FCCeeALLEGRO_gamma_100kEvents_1GeV100GeV_GPSflat_part4.h5"
  "LEMURS_FCCeeALLEGRO_gamma_100kEvents_1GeV100GeV_GPSflat_part5.h5"
  "LEMURS_FCCeeALLEGRO_gamma_100kEvents_1GeV100GeV_GPSflat_part6.h5"
  "LEMURS_FCCeeALLEGRO_gamma_100kEvents_1GeV100GeV_GPSflat_part7.h5"
  "LEMURS_FCCeeALLEGRO_gamma_100kEvents_1GeV100GeV_GPSflat_part8.h5"
  "LEMURS_FCCeeALLEGRO_gamma_100kEvents_1GeV100GeV_GPSflat_part9.h5"
  "LEMURS_FCCeeALLEGRO_gamma_100kEvents_1GeV100GeV_GPSflat_part10.h5"
)

for fname in "${FILES[@]}"; do
  url="${ZENODO}/${fname}/content"
  echo "Downloading $fname ..."
  curl -L -C - --progress-bar -o "$fname" "$url" --retry 3 --retry-delay 15 --connect-timeout 30 &
done
wait
echo "All downloads complete."
