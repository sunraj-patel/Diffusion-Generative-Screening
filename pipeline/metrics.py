"""Predictor evaluation: regression metrics, train-mean baseline, ensemble statistics and uncertainty diagnostics."""
import numpy as np
from scipy.stats import spearmanr, norm


def regression_metrics(y, p, y_train_mean=None):
    y, p = np.asarray(y, float), np.asarray(p, float)
    mae = np.mean(np.abs(y - p)); rmse = np.sqrt(np.mean((y - p) ** 2))
    r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    out = dict(n=len(y), mae=mae, rmse=rmse, r2=r2, y_min=y.min(), y_max=y.max(), y_mean=y.mean())
    if y_train_mean is not None:
        out["baseline_mae"] = np.mean(np.abs(y - y_train_mean)); out["baseline_rmse"] = np.sqrt(np.mean((y - y_train_mean) ** 2))
    return out


def ensemble_stats(preds):
    """preds: (n_members, n_samples). Returns mean and sample std (ddof=1) across members."""
    P = np.asarray(preds, float)
    return P.mean(0), P.std(0, ddof=1) if P.shape[0] > 1 else np.zeros(P.shape[1])


def uncertainty_report(y, mean, std, n_bins=4):
    """Spearman(std, |err|), error per std quantile bin, and a z-score coverage check (calibration evidence)."""
    err = np.abs(y - mean); rho = spearmanr(std, err).correlation
    edges = np.quantile(std, np.linspace(0, 1, n_bins + 1)); idx = np.clip(np.searchsorted(edges, std, side="right") - 1, 0, n_bins - 1)
    bins = [dict(bin=int(b), n=int((idx == b).sum()), mean_std=float(std[idx == b].mean()), mae=float(err[idx == b].mean())) for b in range(n_bins) if (idx == b).any()]
    z = np.abs(y - mean) / np.maximum(std, 1e-9)
    cov = {f"{c:.0%}": float(np.mean(z <= norm.ppf(0.5 + c / 2))) for c in (0.5, 0.8, 0.95)}
    return dict(spearman_std_abs_err=float(rho), bins=bins, gaussian_coverage=cov)
