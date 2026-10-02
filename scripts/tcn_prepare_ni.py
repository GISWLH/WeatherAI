"""Build the directory layout the official TropiCycloneNet loader expects from the public TCND Zenodo record (10.5281/zenodo.15009527, CC-BY-4.0),
North-Indian-Ocean test split only:  <root>/BST_data/NI/test/*.txt, <root>/Env_data/NI/<year>/<TC>/<date>.npy (copied),
<root>/ERA5_gph500/NI/<year>/<TC>/<date>.npy (81x81 geopotential at 500 hPa, converted from TCND_Data3D_NI *.nc, units m2/s2)."""
import glob, os, shutil, sys
import numpy as np
import xarray as xr

SRC = "/workspace/ckpt/tcn/data"
DST = sys.argv[1] if len(sys.argv) > 1 else "/workspace/ckpt/tcn/ni_root"
bst = f"{SRC}/TCND_Data1D/Data1D/NI/test"
os.makedirs(f"{DST}/BST_data/NI/test", exist_ok=True)
kept = []
for f in sorted(os.listdir(bst)):
    year, name = f[2:6], f[:-4].split("BST", 1)[1]
    env = f"{SRC}/TCND_Env-Data/Env-Data/NI/{year}/{name}"
    g3 = f"{SRC}/TCND_Data3D_NI/NI/{year}/{name}"
    if not (os.path.isdir(env) and os.path.isdir(g3)):
        print("skip (no env/3D data):", f)
        continue
    shutil.copy(f"{bst}/{f}", f"{DST}/BST_data/NI/test/{f}")
    os.makedirs(f"{DST}/Env_data/NI/{year}/{name}", exist_ok=True)
    for e in glob.glob(f"{env}/*.npy"):
        shutil.copy(e, f"{DST}/Env_data/NI/{year}/{name}/")
    os.makedirs(f"{DST}/ERA5_gph500/NI/{year}/{name}", exist_ok=True)
    for nc in glob.glob(f"{g3}/*.nc"):
        date = os.path.basename(nc).split("_")[2]
        ds = xr.open_dataset(nc)
        z = ds.z.sel(pressure_level=500).isel(time=0).values.astype(np.float32)
        np.save(f"{DST}/ERA5_gph500/NI/{year}/{name}/{date}.npy", z)
    kept.append(f)
print("kept", len(kept), kept)
