"""Low-dimensional nonnegative envelopes for a fixed periodic carrier.

The envelope and nuisance background use observed pixels only. A 3-by-3
bilinear control grid can express a center-versus-edge amplitude change that
an affine field cannot. It cannot express arbitrarily sharp support boundaries.
No clean reference, true frequency, or true interference mask is accepted.

Bounded linear least squares follows SciPy's documented ``lsq_linear`` API:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.optimize import lsq_linear


@lru_cache(maxsize=32)
def bilinear_basis(shape, grid_size=3):
    """Return a nonnegative partition-of-unity grid on actual pixel centers.

    Control points include the first and last pixel centers, so the amplitude
    bounds hold on every pixel including image edges. The returned cached basis
    is read-only; fitting weights and masks do not modify it.
    """
    if (len(shape) != 2 or any(not isinstance(n, (int, np.integer)) or n < 2 for n in shape)
            or not isinstance(grid_size, (int, np.integer)) or not 2 <= grid_size <= 5
            or grid_size > min(shape)):
        raise ValueError('Expected a 2-D shape >=2 and grid_size in [2,5] within the shape')
    h, w = map(int, shape)

    def hats(length):
        coordinate = np.linspace(0., grid_size-1, length)
        return np.maximum(1.-np.abs(coordinate[:, None]-np.arange(grid_size)), 0.)

    vertical, horizontal = hats(h), hats(w)
    basis = np.einsum('yi,xj->yxij', vertical, horizontal).reshape(h, w, grid_size**2)
    basis.setflags(write=False)
    return basis


@lru_cache(maxsize=4)
def _control_differences(grid_size):
    rows = []
    for y in range(grid_size):
        for x in range(grid_size):
            for dy, dx in ((1, 0), (0, 1)):
                if y+dy < grid_size and x+dx < grid_size:
                    row = np.zeros(grid_size**2)
                    row[y*grid_size+x] = -1.
                    row[(y+dy)*grid_size+x+dx] = 1.
                    rows.append(row)
    result = np.asarray(rows)
    result.setflags(write=False)
    return result


def fit_spatial_envelope(image, carrier, weight, plane, amplitude, *, grid_size=3,
                         smoothness=.002, ridge=.0002, amplitude_cap=2.5):
    """Fit ``Y = carrier * sum(c_j B_j) + plane @ background``, with bounded c.

    The data term is divided by ``sum(weight)``, so changing tile size or a
    scalar weight normalization does not change the regularization strength.
    The objective is data MSE + ``smoothness * mean((D c)**2)`` +
    ``ridge * mean(c**2)``. ``D`` contains neighboring control-point first
    differences. The amplitude ridge shrinks toward zero rather than toward an
    assumed globally present sinusoid, while the smoothness term discourages
    alternating control values. These are fixed coarse experimental settings,
    not learned parameters or noise probabilities.

    All control values are in ``[0, amplitude_cap * amplitude]``; the bilinear
    basis keeps the whole field within those bounds. The default cap matches
    the previous affine-envelope alternative. It may limit narrow components
    whose constant-fit amplitude underestimates their peak amplitude.

    ``sse`` uses original unnormalized weights for compatibility with the
    estimator. Arrays are returned for reconstruction; scalar/list diagnostics
    are JSON-safe. A failed solver raises rather than silently using an
    unconverged envelope. Zero amplitude returns a background-only prediction.
    """
    a = np.asarray(image, dtype=float)
    carrier = np.asarray(carrier, dtype=float)
    weight = np.asarray(weight, dtype=float)
    plane = np.asarray(plane, dtype=float)
    if a.ndim != 2 or min(a.shape) < 2 or not np.isfinite(a).all():
        raise ValueError('image must be a finite 2-D patch, dimensions >=2')
    if carrier.shape != a.shape or not np.isfinite(carrier).all():
        raise ValueError('carrier must be finite and match the image')
    if weight.shape != a.shape or not np.isfinite(weight).all() or np.any(weight < 0):
        raise ValueError('weight must be finite, nonnegative and match the image')
    if plane.ndim != 3 or plane.shape[:2] != a.shape or plane.shape[-1] != 3 or not np.isfinite(plane).all():
        raise ValueError('plane must have shape (height,width,3) with finite values')
    if (not all(np.isfinite(v) for v in (amplitude, smoothness, ridge, amplitude_cap))
            or amplitude < 0 or smoothness < 0 or ridge < 0 or amplitude_cap <= 0):
        raise ValueError('amplitude/regularization must be finite nonnegative and amplitude_cap positive')
    basis = bilinear_basis(a.shape, grid_size)
    mass = float(weight.sum())
    if not np.isfinite(mass) or mass <= 0 or np.count_nonzero(weight) < 3:
        raise ValueError('At least three weighted pixels with positive total weight are required')
    root = np.sqrt(weight.ravel()/mass)
    background_design = plane.reshape(-1, 3)
    if np.linalg.matrix_rank(background_design*root[:, None]) < 3:
        raise ValueError('Weighted background plane must have rank three')
    n_controls = grid_size**2
    differences = _control_differences(grid_size)
    upper = float(amplitude_cap*amplitude)
    if not np.isfinite(upper):
        raise ValueError('The amplitude upper bound must be finite')
    if upper == 0:
        controls = np.zeros(n_controls)
        background = np.linalg.lstsq(background_design*root[:, None], a.ravel()*root, rcond=None)[0]
        status, optimality, iterations = 3, 0., 0
    else:
        design = np.concatenate([carrier[..., None]*basis, plane], axis=-1).reshape(-1, n_controls+3)
        blocks = [design*root[:, None]]
        targets = [a.ravel()*root]
        if smoothness > 0:
            penalty = np.zeros((len(differences), n_controls+3))
            penalty[:, :n_controls] = np.sqrt(smoothness/len(differences))*differences
            blocks.append(penalty); targets.append(np.zeros(len(differences)))
        if ridge > 0:
            penalty = np.zeros((n_controls, n_controls+3))
            penalty[:, :n_controls] = np.sqrt(ridge/n_controls)*np.eye(n_controls)
            blocks.append(penalty); targets.append(np.zeros(n_controls))
        fit = lsq_linear(np.vstack(blocks), np.concatenate(targets),
            bounds=(np.r_[np.zeros(n_controls), np.full(3, -np.inf)],
                    np.r_[np.full(n_controls, upper), np.full(3, np.inf)]),
            method='bvls', tol=1e-10, max_iter=100)
        if not fit.success or not np.isfinite(fit.x).all():
            raise RuntimeError(f'Nonnegative envelope fit did not converge: {fit.message}')
        controls = np.clip(fit.x[:n_controls], 0., upper)
        background = fit.x[n_controls:]
        status, optimality, iterations = int(fit.status), float(fit.optimality), int(fit.nit)
    field = basis@controls
    residual = field*carrier
    prediction = residual+plane@background
    sse = float(np.sum(weight*(a-prediction)**2))
    smooth_penalty = float(smoothness*np.mean((differences@controls)**2))
    amplitude_penalty = float(ridge*np.mean(controls**2))
    return dict(field=field, residual=residual, prediction=prediction, sse=sse,
                coefficients=controls.reshape(grid_size, grid_size).tolist(),
                background_coefficients=background.tolist(), grid_size=int(grid_size),
                data_mse=sse/mass, smoothness_penalty=smooth_penalty,
                amplitude_penalty=amplitude_penalty,
                objective=sse/mass+smooth_penalty+amplitude_penalty,
                amplitude_upper=upper, solver_status=status,
                solver_optimality=optimality, solver_iterations=iterations)


__all__ = ['bilinear_basis', 'fit_spatial_envelope']
