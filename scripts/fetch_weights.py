"""Generic weight fetchers for the HF Space (run on the Space CPU outside the 120 s ZeroGPU window). Weights never enter git."""
from __future__ import annotations

import os


def fetch_stormcast(out: str, log=print) -> None:
    """nvidia/stormcast-v1-era5-hrrr (Apache-2.0, ungated): the two .mdlus networks (~800 MB) + normalisation metadata."""
    from weatherai.models.stormcast.convert import download

    log("downloading nvidia/stormcast-v1-era5-hrrr ...")
    download(out)
    log({f: os.path.getsize(os.path.join(out, f)) for f in sorted(os.listdir(out)) if os.path.isfile(os.path.join(out, f))})


def fetch_arches(out: str, log=print) -> None:
    """HF gcouairon/ArchesWeather (BSD): ArchesWeatherGen checkpoint (1.9 GB, embeds the 4 deterministic members) + det seed0, geoarches stats
    files (BSD, from GitHub) and a 4-step WeatherBench2 ERA5 sample (public GCS) for the real-case check."""
    from weatherai.models.arches.convert import download

    log("downloading gcouairon/ArchesWeather + geoarches stats ...")
    download(out)
    try:
        import subprocess, sys

        subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "fetch_wb2_sample.py"), os.path.join(out, "wb2_sample_2020.nc")], check=True)
    except Exception as e:  # optional
        log(f"WB2 sample not fetched: {e!r}")
    log({f: os.path.getsize(os.path.join(out, f)) for f in sorted(os.listdir(out)) if os.path.isfile(os.path.join(out, f))})


def fetch_ace2(out: str, log=print) -> None:
    """HF allenai/ACE2-ERA5 (Apache-2.0): checkpoint 1.8 GB, 2020 initial conditions and forcing (0.55 GB), plus a WeatherBench2 ERA5 sample."""
    from weatherai.models.ace2.convert import download

    log("downloading allenai/ACE2-ERA5 ...")
    download(out)
    try:
        import subprocess, sys

        subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "fetch_wb2_ace2.py"), os.path.join(out, "wb2_sample_2020.nc")], check=True)
    except Exception as e:  # optional
        log(f"WB2 sample not fetched: {e!r}")
    log({f: os.path.getsize(os.path.join(out, f)) for f in sorted(os.listdir(out)) if os.path.isfile(os.path.join(out, f))})


def fetch_orca_dl(out: str, log=print) -> None:
    """HF dataset JayKuo/ORCA-DL-data (licence not stated): seed_1.bin 2.16 GB + monthly statistics, plus the official demo input from GitHub."""
    from weatherai.models.orca_dl import download, download_example

    log("downloading JayKuo/ORCA-DL-data seed_1 + stat ...")
    download(out, seeds=(1,))
    download_example(os.path.join(out, "example"))
    log("done")


FETCHERS = {"orca_dl": fetch_orca_dl, "ace2": fetch_ace2, "stormcast": fetch_stormcast, "arches": fetch_arches}
