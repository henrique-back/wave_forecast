"""
Fetch NOAA GEFSv12 wave reforecast point spectra at one NDBC station, for the
physical-model baseline in scripts/compare_physical_baseline.py.

Source: the public GEFSv12 wave reforecast (WAVEWATCH III; Campos et al. 2024,
Wea. Forecasting 39:1651-1672) on AWS,
    https://noaa-nws-gefswaves-reforecast-pds.s3.amazonaws.com/GEFSv12/reforecast/
    YYYY/YYYYMMDD/station/gefs.wave.YYYYMMDD.<member>.{spec,tab}.nc
One 00Z cycle per day, 16-day range. The spectral file holds 3-hourly 2-D
spectra (efth, m2 s rad-1; 50 frequencies 0.035-0.963 Hz x 36 directions) for
658 points, valid from +3 h to +384 h; there is no lead-0 output. The wave
model assimilates no wave data (each cycle starts from the previous cycle's
24 h forecast), so the forecast is independent of the buoy it is scored
against.

The daily spectral file is ~516 MB, but efth is chunked per time step and per
half of the stations, so only the chunks holding this station and leads up to
MAX_LEAD_H are read, via HTTP Range requests (~32 MB per cycle for 48 h,
roughly twice that for 96 h); the
table file is read the same way. Each request goes through curl with a hard
deadline, because on a poor connection a single stalled request can otherwise
hang for tens of minutes. The spectrum is integrated over
direction (sum x 2*pi/36 rad) and the 1-D result is cached per date under
downloads/gefsv12/<station>/ (git-ignored; re-running skips cached dates whose
cache reaches MAX_LEAD_H, and re-fetches shorter ones).
Extracted Hs = 4*sqrt(sum E(f) df), with df from the file's own band edges,
must match the table file's Hs to within HS_TOL. assemble() then writes one
small file, buoy_data/<station>/gefsv12_<member>_spec1d.npz:
    init_times (cycles,) datetime64[ns]   00Z cycle times
    lead_hours (leads,)                   3, 6, ..., MAX_LEAD_H
    freqs, freq_lo, freq_hi (F,)          band centres and edges [Hz]
    E1d (cycles, leads, F)                1-D spectrum [m2/Hz]
    hs_tab (cycles, leads)                table-file Hs at the same times [m]
    lat, lon                              model point position

Run manually (needs internet; ~110 cycles for the default window):
    python scripts/fetch_gefs_reforecast.py
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import io
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

STATION = "32012"
MEMBER = "c00"                      # control member only
NOMINAL_LAT, NOMINAL_LON = -19.425, -85.078   # buoy_data/32012/buoy_ID.txt
MAX_POS_ERR_DEG = 0.1
START_DATE, END_DATE = "2017-09-13", "2017-12-31"   # the test split
MAX_LEAD_H = 96                     # longest lead scored; more leads read more chunks
                                    # (48 for compare_physical_baseline.py alone, 96 for
                                    # the lead-time study, scripts/eval_lead_curves.py)
N_WORKERS = 8                       # requests are latency-bound, not bandwidth-bound
N_RETRIES = 5
REQUEST_TIMEOUT_S = 30              # hard deadline per range request
HS_TOL = 0.01                       # relative, extracted vs table Hs
BASE_URL = "https://noaa-nws-gefswaves-reforecast-pds.s3.amazonaws.com/GEFSv12/reforecast"

project_root = Path(__file__).resolve().parent.parent
cache_dir = project_root / "downloads" / "gefsv12" / STATION
out_path = project_root / "buoy_data" / STATION / f"gefsv12_{MEMBER}_spec1d.npz"


def decode_times(days_since_1990):
    """WW3 'days since 1990-01-01' floats -> tz-naive UTC DatetimeIndex on whole minutes."""
    times = pd.Timestamp("1990-01-01") + pd.to_timedelta(np.asarray(days_since_1990), unit="D")
    rounded = times.round("min")
    if np.abs((times - rounded).total_seconds()).max() > 1.0:
        raise ValueError("WW3 times are not on whole minutes")
    return pd.DatetimeIndex(rounded)


def integrate_direction(efth, n_dirs):
    """(..., F, D) directional spectrum per radian -> (..., F) 1-D spectrum."""
    return efth.sum(axis=-1) * (2 * np.pi / n_dirs)


def _station_index(ds, station):
    names = [b"".join(row).decode().strip() for row in ds["station_name"][:]]
    return names.index(station)


def _with_retries(fn, what):
    for attempt in range(1, N_RETRIES + 1):
        try:
            return fn()
        except Exception as err:
            if attempt == N_RETRIES:
                raise
            print(f"  retry {attempt} for {what}: {err}")
            time.sleep(10 * attempt)


def _curl(args):
    """Run curl with a hard deadline (a stalled connection can otherwise
    trickle on indefinitely) and return stdout bytes."""
    out = subprocess.run(["curl", "-sSf", "--max-time", str(REQUEST_TIMEOUT_S), *args],
                         capture_output=True)
    if out.returncode != 0:
        raise IOError(f"curl exit {out.returncode}: {out.stderr.decode().strip()}")
    return out.stdout


class HTTPRangeFile(io.RawIOBase):
    """Read-only, seekable remote file over HTTP Range requests, for h5py.File.

    Reads smaller than BLOCK (HDF5 metadata is many tiny reads) are served
    from cached BLOCK-aligned blocks; larger reads (data chunks) go straight
    through as one request.
    """
    BLOCK = 1 << 18

    def __init__(self, url):
        super().__init__()
        self.url, self.pos, self._blocks = url, 0, {}
        headers = _with_retries(lambda: _curl(["-I", url]).decode(), url)
        self.size = int(next(line.split(":", 1)[1] for line in headers.splitlines()
                             if line.lower().startswith("content-length")))

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: offset, io.SEEK_CUR: self.pos + offset,
                    io.SEEK_END: self.size + offset}[whence]
        return self.pos

    def _get(self, start, end):
        def once():
            data = _curl(["-r", f"{start}-{end - 1}", self.url])
            if len(data) != end - start:
                raise IOError(f"range {start}-{end}: got {len(data)} bytes")
            return data
        return _with_retries(once, f"{self.url} [{start}:{end}]")

    def _block(self, i):
        if i not in self._blocks:
            self._blocks[i] = self._get(i * self.BLOCK, min((i + 1) * self.BLOCK, self.size))
        return self._blocks[i]

    def readinto(self, buf):
        n = min(len(buf), self.size - self.pos)
        if n <= 0:
            return 0
        if n >= self.BLOCK:
            data = self._get(self.pos, self.pos + n)
        else:
            first, last = self.pos // self.BLOCK, (self.pos + n - 1) // self.BLOCK
            joined = b"".join(self._block(i) for i in range(first, last + 1))
            offset = self.pos - first * self.BLOCK
            data = joined[offset:offset + n]
        buf[:n] = data
        self.pos += n
        return n


def cache_ok(date):
    """True if this date is cached with leads reaching MAX_LEAD_H."""
    path = cache_dir / f"{date:%Y%m%d}.npz"
    if not path.exists():
        return False
    valid = pd.DatetimeIndex(np.load(path)["valid_times"])
    return (valid[-1] - date) / pd.Timedelta(hours=1) >= MAX_LEAD_H


def fetch_date(date):
    """Extract and cache one cycle; returns the cache path."""
    cache_path = cache_dir / f"{date:%Y%m%d}.npz"
    if cache_ok(date):
        return cache_path
    stem = f"{BASE_URL}/{date:%Y}/{date:%Y%m%d}/station/gefs.wave.{date:%Y%m%d}.{MEMBER}"

    with h5py.File(HTTPRangeFile(f"{stem}.spec.nc"), "r") as spec:
        i = _station_index(spec, STATION)
        valid_all = decode_times(spec["time"][:])
        leads_all = (valid_all - date) / pd.Timedelta(hours=1)
        n_keep = int(np.sum(np.asarray(leads_all) <= MAX_LEAD_H))
        if not np.array_equal(np.asarray(leads_all[:n_keep]), np.arange(3, MAX_LEAD_H + 1, 3)):
            raise ValueError(f"{date:%Y%m%d}: unexpected lead times {list(leads_all[:n_keep])}")
        valid = valid_all[:n_keep]
        lat, lon = float(spec["latitude"][0, i]), float(spec["longitude"][0, i])
        if abs(lat - NOMINAL_LAT) > MAX_POS_ERR_DEG or abs(lon - NOMINAL_LON) > MAX_POS_ERR_DEG:
            raise ValueError(f"{date:%Y%m%d}: station {STATION} at ({lat}, {lon})")
        freqs = spec["frequency"][:].astype(np.float64)
        freq_lo = spec["frequency1"][:].astype(np.float64)
        freq_hi = spec["frequency2"][:].astype(np.float64)
        efth = spec["efth"][:n_keep, i, :, :].astype(np.float64)       # (time, F, D)
    e1d = integrate_direction(efth, efth.shape[-1])

    with h5py.File(HTTPRangeFile(f"{stem}.tab.nc"), "r") as tab:
        it = _station_index(tab, STATION)
        tab_times = decode_times(tab["time"][:])
        n_tab = int(np.searchsorted(tab_times, valid[-1], side="right"))
        hs_tab = (pd.Series(tab["hs"][:n_tab, it].astype(np.float64), index=tab_times[:n_tab])
                  .reindex(valid).to_numpy())

    hs = 4 * np.sqrt((e1d * (freq_hi - freq_lo)).sum(axis=1))
    rel = np.abs(hs / hs_tab - 1)
    if not np.all(rel < HS_TOL):
        raise ValueError(f"{date:%Y%m%d}: extracted Hs differs from table Hs by up to {np.nanmax(rel):.3f}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, valid_times=valid.to_numpy(), e1d=e1d, hs_tab=hs_tab, freqs=freqs,
             freq_lo=freq_lo, freq_hi=freq_hi, lat=lat, lon=lon)
    return cache_path


def assemble(dates):
    """Stack the cached cycles into one (cycles, leads, F) file."""
    init_times, e1d, hs_tab, lead_hours = [], [], [], None
    for date in dates:
        c = np.load(cache_dir / f"{date:%Y%m%d}.npz")
        leads = np.asarray((pd.DatetimeIndex(c["valid_times"]) - date) / pd.Timedelta(hours=1), dtype=float)
        keep = leads <= MAX_LEAD_H          # a cache fetched for a longer lead is truncated
        leads = leads[keep]
        if lead_hours is None:
            lead_hours, freqs, freq_lo, freq_hi = leads, c["freqs"], c["freq_lo"], c["freq_hi"]
            lat, lon = float(c["lat"]), float(c["lon"])
        elif not (np.array_equal(leads, lead_hours) and np.array_equal(c["freqs"], freqs)):
            raise ValueError(f"{date:%Y%m%d}: lead times or frequencies differ from the first cycle")
        init_times.append(date)
        e1d.append(c["e1d"][keep])
        hs_tab.append(c["hs_tab"][keep])
    e1d, hs_tab = np.stack(e1d), np.stack(hs_tab)
    if not np.isfinite(e1d).all():
        raise ValueError("non-finite values in the assembled spectra")
    np.savez_compressed(out_path, init_times=pd.DatetimeIndex(init_times).to_numpy(),
                        lead_hours=lead_hours, freqs=freqs, freq_lo=freq_lo, freq_hi=freq_hi,
                        E1d=e1d.astype(np.float32), hs_tab=hs_tab.astype(np.float32),
                        lat=lat, lon=lon)
    print(f"Wrote {out_path}: {len(init_times)} cycles x {len(lead_hours)} leads x {len(freqs)} freqs")


def main():
    dates = list(pd.date_range(START_DATE, END_DATE, freq="D"))
    todo = [d for d in dates if not cache_ok(d)]
    print(f"{len(dates)} cycles, {len(dates) - len(todo)} already cached, fetching {len(todo)}")
    failed = []
    # Processes, not threads: h5py holds one global lock around every HDF5
    # call, including the ranged reads it makes through HTTPRangeFile.
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(fetch_date, d): d for d in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            d = futures[fut]
            try:
                fut.result()
                print(f"  [{n}/{len(todo)}] {d:%Y-%m-%d} ok")
            except Exception as err:
                failed.append(d)
                print(f"  [{n}/{len(todo)}] {d:%Y-%m-%d} FAILED: {err}")
    if failed:
        print(f"{len(failed)} cycles failed; re-run to retry: {[f'{d:%Y%m%d}' for d in failed]}")
        return
    assemble(dates)


if __name__ == "__main__":
    main()
