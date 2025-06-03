import numpy as np
from typing import Literal, Tuple
from scipy.interpolate import CloughTocher2DInterpolator
import pykeops.torch as keops
import torch


def interpolate_to_grid(
    method: Literal["knn", "ct", "idw", "gaussian"],
    x_train: np.ndarray,
    z_train: np.ndarray,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    resolution: float,
    k: int = 4,
    gamma: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate scattered data to a regular grid.

    Parameters
    ----------
    method : Interpolation method: "knn", "ct" (CloughTocher), "idw", or "gaussian".
    x_train : (N, 2) input coordinates.
    z_train : (N,) input values.
    xlim : (xmin, xmax)
    ylim : (ymin, ymax)
    resolution : size of one pixel (in same unit as x/y)
    k : number of neighbors for knn, idw, gaussian
    gamma : bandwidth for gaussian kernel (used only in gaussian mode)

    Returns
    -------
    z_grid : interpolated values on the grid
    X, Y : coordinate grids
    """
    x_train = np.asarray(x_train)
    z_train = np.asarray(z_train)
    
    if x_train.ndim != 2 or x_train.shape[1] != 2:
        raise ValueError("x_train must be of shape (N, 2)")

    # Generate regular grid
    x_vals = np.arange(xlim[0], xlim[1] + resolution, resolution)
    y_vals = np.arange(ylim[0], ylim[1] + resolution, resolution)
    X, Y = np.meshgrid(x_vals, y_vals)
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)

    if method == "ct":
        interpolator = CloughTocher2DInterpolator(x_train, z_train)
        z_grid = interpolator(grid_points).reshape(X.shape)
        return z_grid, X, Y

    # PyKeOps methods
    x_i = torch.tensor(grid_points, dtype=torch.float32)
    x_j = torch.tensor(x_train, dtype=torch.float32)
    z_j = torch.tensor(z_train.reshape(-1, 1), dtype=torch.float32)

    if method == "knn":
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(x_train)
        dist, idx = nn.kneighbors(grid_points)
        weights = 1.0 / (dist + 1e-12)
        weights /= weights.sum(axis=1, keepdims=True)
        z_grid = np.sum(z_train[idx] * weights, axis=1).reshape(X.shape)
        return z_grid, X, Y

    elif method == "idw":
        D = ((x_i[:, None, :] - x_j[None, :, :])**2).sum(dim=2).sqrt()
        weights = 1.0 / (D + 1e-6)
        weights /= weights.sum(dim=1, keepdim=True)
        z_pred = (weights @ z_j).squeeze().numpy().reshape(X.shape)
        return z_pred, X, Y

    elif method == "gaussian":
        D2 = ((x_i[:, None, :] - x_j[None, :, :])**2).sum(dim=2)
        weights = torch.exp(-D2 / (2 * gamma**2))
        weights /= weights.sum(dim=1, keepdim=True)
        z_pred = (weights @ z_j).squeeze().numpy().reshape(X.shape)
        return z_pred, X, Y

    else:
        raise ValueError(f"Unknown interpolation method: {method}")
