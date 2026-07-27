"""Combine parts 1-9 into LEMURS_FCCeeALLEGRO_training.h5 (no compression for speed, then repack)."""
import sys
import h5py
import numpy as np
from pathlib import Path

base_dir = Path("../lemurs/fcceeallegro")
output_path = base_dir / "LEMURS_FCCeeALLEGRO_training.h5"
part_files = sorted(base_dir.glob(
    "LEMURS_FCCeeALLEGRO_gamma_100kEvents_1GeV100GeV_GPSflat_part[1-9].h5"
))

print(f"Combining {len(part_files)} files", flush=True)

# Inspect first file
with h5py.File(part_files[0], "r") as f0:
    keys = list(f0.keys())
    dtypes = {k: f0[k].dtype for k in keys}
    scalar_keys = [k for k in keys if f0[k].ndim == 1]
    array_keys = [k for k in keys if f0[k].ndim > 1]
    array_shape_suffix = {k: f0[k].shape[1:] for k in array_keys}

# Phase 1: compute total events
total_events = 0
part_events = []
for fpath in part_files:
    with h5py.File(fpath, "r") as f:
        n = f[scalar_keys[0]].shape[0]
        part_events.append(n)
        total_events += n
    print(f"  {fpath.name}: {n} events", flush=True)

print(f"Total events: {total_events}", flush=True)

CHUNK_SIZE = 5000

# Phase 2: create output (NO compression for speed)
with h5py.File(output_path, "w") as out_f:
    out_ds = {}
    for key in keys:
        suffix = array_shape_suffix.get(key, ())
        chunk_shape = (min(CHUNK_SIZE, total_events),) + suffix
        ds = out_f.create_dataset(
            key,
            shape=(0,) + suffix,
            maxshape=(None,) + suffix,
            dtype=dtypes[key],
            chunks=chunk_shape,
            # no compression
        )
        out_ds[key] = ds

    # Phase 3: stream each part
    for part_idx, (fpath, n_evts) in enumerate(zip(part_files, part_events)):
        sys.stdout.write(f"[{part_idx+1}/{len(part_files)}] {fpath.name} ({n_evts} events)")
        sys.stdout.flush()

        with h5py.File(fpath, "r") as in_f:
            for start in range(0, n_evts, CHUNK_SIZE):
                end = min(start + CHUNK_SIZE, n_evts)
                chunk_data = {}
                for k in keys:
                    chunk_data[k] = in_f[k][start:end]

                for k in keys:
                    ds = out_ds[k]
                    cur = ds.shape[0]
                    ds.resize(cur + chunk_data[k].shape[0], axis=0)
                    ds[cur:] = chunk_data[k]

            # Flush after each part to ensure data is on disk
            out_f.flush()

        print(" ✓", flush=True)

# Phase 4: report
print(f"\nDone: {output_path}", flush=True)
with h5py.File(output_path, "r") as f:
    for key in keys:
        print(f"  {key}: {f[key].shape} (dtype={f[key].dtype})", flush=True)
print(f"Size: {output_path.stat().st_size / 1e9:.2f} GB", flush=True)
