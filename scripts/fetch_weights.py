"""Generic weight fetchers for the HF Space (run on the Space CPU outside the 120 s ZeroGPU window). Weights never enter git."""
from __future__ import annotations

import os


def fetch_stormcast(out: str, log=print) -> None:
    """nvidia/stormcast-v1-era5-hrrr (Apache-2.0, ungated): the two .mdlus networks (~800 MB) + normalisation metadata."""
    from weatherai.models.stormcast.convert import download

    log("downloading nvidia/stormcast-v1-era5-hrrr ...")
    download(out)
    log({f: os.path.getsize(os.path.join(out, f)) for f in sorted(os.listdir(out)) if os.path.isfile(os.path.join(out, f))})


FETCHERS = {"stormcast": fetch_stormcast}
