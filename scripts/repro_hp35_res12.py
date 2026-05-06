import numpy as np
from bvvmmm import SineBVvMMM
from pathlib import Path

HERE = Path(__file__).resolve().parent
X = np.load(HERE / "repro_macro3_res12_phi_psi.npy")

m = SineBVvMMM(n_components=5, verbose=True, init_method="random", seed=1234)
m.fit(X)
print("after fit:", m.weights_, m.means_, m.kappas_)

m.refine(X)
print("after refine:", m.weights_, m.means_, m.kappas_)

assert np.all(np.isfinite(m.weights_.detach().cpu().numpy()))
assert np.all(np.isfinite(m.means_.detach().cpu().numpy()))
assert np.all(np.isfinite(m.kappas_.detach().cpu().numpy()))
