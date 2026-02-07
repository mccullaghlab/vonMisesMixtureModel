import numpy as np

def wrap_to_pi(x):
    """
    Wrap angles to [-pi, pi).
    Works with scalars or numpy arrays.
    """
    return (x + np.pi) % (2 * np.pi) - np.pi


def gibbs_sine_bvm(
    n_samples,
    mu1,
    mu2,
    kappa1,
    kappa2,
    lam,
    burn_in=1000,
    thin=1,
    init=None,
    rng=None,
    return_raw_xy=False,
):
    """
    Gibbs sampler for the *sine* bivariate von Mises distribution:

        p(phi, psi) ∝ exp(
            kappa1*cos(phi-mu1) +
            kappa2*cos(psi-mu2) +
            lam*sin(phi-mu1)*sin(psi-mu2)
        ),  with phi, psi ∈ [-pi, pi).

    Uses the fact that the conditionals are univariate von Mises:
      x = phi - mu1, y = psi - mu2
      x | y ~ VM(mean=atan2(lam*sin y, kappa1), kappa=sqrt(kappa1^2 + (lam*sin y)^2))
      y | x ~ VM(mean=atan2(lam*sin x, kappa2), kappa=sqrt(kappa2^2 + (lam*sin x)^2))

    Parameters
    ----------
    n_samples : int
        Number of returned (post burn-in, post thinning) samples.
    mu1, mu2 : float
        Mean directions (radians).
    kappa1, kappa2 : float
        Concentrations (>= 0).
    lam : float
        Coupling parameter (can be positive or negative).
    burn_in : int
        Number of initial Gibbs steps discarded.
    thin : int
        Keep one sample every `thin` Gibbs steps.
    init : tuple or None
        Optional initial (phi, psi). If None, starts at (mu1, mu2).
    rng : np.random.Generator or None
        Random generator. If None, uses default_rng().
    return_raw_xy : bool
        If True, also returns (x, y) in centered coordinates.

    Returns
    -------
    samples : (n_samples, 2) ndarray
        Columns are (phi, psi), wrapped to [-pi, pi).
    (optional) raw_xy : (n_samples, 2) ndarray
        Columns are (x, y) = (phi-mu1, psi-mu2) wrapped to [-pi, pi).
    """
    if rng is None:
        rng = np.random.default_rng()

    if kappa1 < 0 or kappa2 < 0:
        raise ValueError("kappa1 and kappa2 must be >= 0")

    total_kept = n_samples
    total_steps = burn_in + total_kept * thin

    # Initialize centered coords x, y
    if init is None:
        x = 0.0
        y = 0.0
    else:
        phi0, psi0 = init
        x = wrap_to_pi(phi0 - mu1)
        y = wrap_to_pi(psi0 - mu2)

    out = np.empty((n_samples, 2), dtype=float)
    if return_raw_xy:
        out_xy = np.empty((n_samples, 2), dtype=float)

    kept = 0
    for t in range(total_steps):
        # --- Sample x | y
        siny = np.sin(y)
        b1 = lam * siny
        R1 = np.hypot(kappa1, b1)              # sqrt(kappa1^2 + b1^2)
        delta1 = np.arctan2(b1, kappa1)        # mean shift
        # numpy's vonmises takes (mu, kappa)
        x = rng.vonmises(delta1, R1)
        x = wrap_to_pi(x)

        # --- Sample y | x
        sinx = np.sin(x)
        b2 = lam * sinx
        R2 = np.hypot(kappa2, b2)
        delta2 = np.arctan2(b2, kappa2)
        y = rng.vonmises(delta2, R2)
        y = wrap_to_pi(y)

        # --- Record (after burn-in, with thinning)
        if t >= burn_in and ((t - burn_in) % thin == 0):
            phi = wrap_to_pi(mu1 + x)
            psi = wrap_to_pi(mu2 + y)
            out[kept, 0] = phi
            out[kept, 1] = psi
            if return_raw_xy:
                out_xy[kept, 0] = x
                out_xy[kept, 1] = y
            kept += 1
            if kept == n_samples:
                break

    return (out, out_xy) if return_raw_xy else out


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    mu1, mu2 = -1.0, 1.0
    k1, k2 = 7.4, 13.3
    lam = -3.7

    rng = np.random.default_rng(0)
    samples = gibbs_sine_bvm(
        n_samples=50_000,
        mu1=mu1, mu2=mu2,
        kappa1=k1, kappa2=k2,
        lam=lam,
        burn_in=2_000,
        thin=1,
        rng=rng
    )

    # quick sanity checks
    print("phi range:", samples[:,0].min(), samples[:,0].max())
    print("psi range:", samples[:,1].min(), samples[:,1].max())
    print("mean phi (circular):", np.angle(np.mean(np.exp(1j*samples[:,0]))))
    print("mean psi (circular):", np.angle(np.mean(np.exp(1j*samples[:,1]))))

