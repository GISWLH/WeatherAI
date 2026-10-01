"""Generate tests/models/neuralgcm/data/sht_ref.npz from official dinosaur (SHT + spectral operators on the 2.8 deg grid)."""
import numpy as np
import jax
import jax.numpy as jnp
from dinosaur import spherical_harmonic as sh
import neuralgcm, pickle

ck = pickle.load(open("/workspace/ckpt/ngcm_det_2_8.pkl", "rb"))
m = neuralgcm.PressureLevelModel.from_checkpoint(ck)
g = m.model_coords.horizontal
rng = np.random.RandomState(0)
mask = np.asarray(g.mask)
modal = rng.randn(2, *g.modal_shape).astype(np.float32) * mask
nodal = rng.randn(2, *g.nodal_shape).astype(np.float32)
u = rng.randn(2, *g.nodal_shape).astype(np.float32) * 10
v = rng.randn(2, *g.nodal_shape).astype(np.float32) * 10
out = dict(modal=modal, nodal=nodal, u=u, v=v)
out["to_nodal"] = np.asarray(g.to_nodal(jnp.asarray(modal)))
out["to_modal"] = np.asarray(g.to_modal(jnp.asarray(nodal)))
out["lap"] = np.asarray(g.laplacian(jnp.asarray(modal)))
out["inv_lap"] = np.asarray(g.inverse_laplacian(jnp.asarray(modal)))
out["d_dlon"] = np.asarray(g.d_dlon(jnp.asarray(modal)))
out["cos_lat_d_dlat"] = np.asarray(g.cos_lat_d_dlat(jnp.asarray(modal)))
out["sec_lat_d_dlat_cos2"] = np.asarray(g.sec_lat_d_dlat_cos2(jnp.asarray(modal)))
gr = g.cos_lat_grad(jnp.asarray(modal)); out["grad0"], out["grad1"] = map(np.asarray, gr)
out["div"] = np.asarray(g.div_cos_lat((jnp.asarray(modal), jnp.asarray(modal[::-1].copy()))))
out["curl"] = np.asarray(g.curl_cos_lat((jnp.asarray(modal), jnp.asarray(modal[::-1].copy()))))
vor, div = sh.uv_nodal_to_vor_div_modal(g, jnp.asarray(u), jnp.asarray(v))
out["vor"], out["div_uv"] = np.asarray(vor), np.asarray(div)
u2, v2 = sh.vor_div_to_uv_nodal(g, vor, div)
out["u_rt"], out["v_rt"] = np.asarray(u2), np.asarray(v2)
out["lat"] = np.asarray(g.latitudes); out["lon"] = np.asarray(g.longitudes)
# orography: official base modal orography (FilteredCustomOrography) for comparison
print({k: v.shape for k, v in out.items()})
np.savez_compressed("tests/models/neuralgcm/data/sht_ref.npz", **out)
