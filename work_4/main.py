from __future__ import annotations

from io import StringIO
from pathlib import Path
import json
import re

import matplotlib
if "__file__" in globals():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if (ROOT / "work_4").is_dir():
    ROOT = ROOT / "work_4"
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)
STATION = "NAIN"
YEAR = 2015


def nmdb_url(start: pd.Timestamp, end: pd.Timestamp) -> str:
    params = {
        "formchk": 1,
        "stations[]": STATION,
        "tabchoice": "revori",
        "dtype": "corr_for_efficiency",
        "tresolution": 0,
        "yunits": 0,
        "date_choice": "bydate",
        "start_day": start.day,
        "start_month": start.month,
        "start_year": start.year,
        "start_hour": start.hour,
        "start_min": start.minute,
        "end_day": end.day,
        "end_month": end.month,
        "end_year": end.year,
        "end_hour": end.hour,
        "end_min": end.minute,
        "output": "ascii",
    }
    response = requests.get(
        "https://www.nmdb.eu/nest/draw_graph.php", params=params, timeout=180
    )
    response.raise_for_status()
    return response.text


def parse_nmdb(text: str) -> pd.Series:
    lines = [
        line.strip()
        for line in text.splitlines()
        if re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}.*;", line.strip())
    ]
    if not lines:
        raise ValueError("В ответе NMDB не найдены строки измерений")
    table = pd.read_csv(StringIO("\n".join(lines)), sep=";", header=None)
    index = pd.to_datetime(table[0], utc=True)
    values = pd.to_numeric(table[1], errors="coerce")
    return pd.Series(values.to_numpy(float), index=index).sort_index()


def load_year() -> pd.Series:
    cache = DATA_DIR / f"{STATION}_{YEAR}_30min.txt"
    if not cache.exists():
        start = pd.Timestamp(f"{YEAR}-01-01 00:00", tz="UTC")
        end = pd.Timestamp(f"{YEAR}-12-31 23:59", tz="UTC")
        cache.write_text(nmdb_url(start, end), encoding="utf-8")
    return parse_nmdb(cache.read_text(encoding="utf-8"))


def daily_series(series: pd.Series) -> pd.Series:
    daily = series.resample("1D").mean()
    full_index = pd.date_range(f"{YEAR}-01-01", f"{YEAR}-12-31", freq="D", tz="UTC")
    daily = daily.reindex(full_index)
    return daily.interpolate(limit_direction="both")


def design_matrix(t: np.ndarray, period: float) -> np.ndarray:
    angle = 2 * np.pi * t / period
    return np.column_stack([np.ones_like(t), t, np.sin(angle), np.cos(angle)])


def fit_period(t: np.ndarray, y: np.ndarray, period: float) -> tuple[np.ndarray, float, np.ndarray]:
    matrix = design_matrix(t, period)
    beta, *_ = np.linalg.lstsq(matrix, y, rcond=None)
    prediction = matrix @ beta
    return beta, float(np.mean((y - prediction) ** 2)), prediction


def golden_period(t: np.ndarray, y: np.ndarray, initial: float) -> tuple[float, np.ndarray, float, np.ndarray]:
    lower = max(2.0, initial / 2.0)
    upper = min(500.0, initial * 2.0)
    grid = np.geomspace(lower, upper, 160)
    losses = np.array([fit_period(t, y, p)[1] for p in grid])
    best_idx = int(losses.argmin())
    left = grid[max(0, best_idx - 1)]
    right = grid[min(len(grid) - 1, best_idx + 1)]
    if left == right:
        left, right = lower, upper
    phi = (1 + np.sqrt(5)) / 2
    c = right - (right - left) / phi
    d = left + (right - left) / phi
    fc = fit_period(t, y, c)[1]
    fd = fit_period(t, y, d)[1]
    for _ in range(80):
        if fc < fd:
            right, d, fd = d, c, fc
            c = right - (right - left) / phi
            fc = fit_period(t, y, c)[1]
        else:
            left, c, fc = c, d, fd
            d = left + (right - left) / phi
            fd = fit_period(t, y, d)[1]
    period = (left + right) / 2
    beta, loss, prediction = fit_period(t, y, period)
    return period, beta, loss, prediction


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = y - prediction
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "RMSE": float(np.sqrt(np.mean(residual**2))),
        "MAE": float(np.mean(np.abs(residual))),
        "R2": 1 - ss_res / ss_tot,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std(ddof=0)),
    }


def main(show_plots: bool = False) -> dict:
    raw = load_year()
    daily = daily_series(raw)
    t = np.arange(len(daily), dtype=float)
    y = daily.to_numpy(float)
    initial_periods = [7.0, 27.0, 90.0, 180.0]
    rows = []
    fits = {}
    for initial in initial_periods:
        period, beta, loss, prediction = golden_period(t, y, initial)
        row = {"initial_period": initial, "period": period, **metrics(y, prediction)}
        rows.append(row)
        fits[initial] = (period, beta, prediction)
    comparison = pd.DataFrame(rows).sort_values(["RMSE", "initial_period"]).reset_index(drop=True)
    best_initial = float(comparison.iloc[0]["initial_period"])
    best_period, best_beta, best_prediction = fits[best_initial]
    residual = y - best_prediction
    result = {
        "station": STATION,
        "year": YEAR,
        "raw_observations": int(raw.notna().sum()),
        "days": int(len(daily)),
        "missing_after_interpolation": int(daily.isna().sum()),
        "best_initial_period": best_initial,
        "best_period": float(best_period),
        "coefficients": best_beta.tolist(),
        "metrics": metrics(y, best_prediction),
    }
    comparison.to_csv(DATA_DIR / "model_comparison.csv", index=False)
    daily.rename("daily_mean").to_csv(DATA_DIR / "daily_series.csv", header=True)
    (DATA_DIR / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(daily.index, y, color="0.35", linewidth=0.8, label="Суточное среднее NAIN")
    ax.plot(daily.index, best_prediction, color="tab:red", linewidth=1.5, label=f"Модель, P = {best_period:.1f} суток")
    ax.set(title="Нелинейная регрессия суточного ряда NAIN за 2015 год", xlabel="Дата UTC", ylabel="RCORR_E")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "regression_fit.png", dpi=180)
    if show_plots:
        plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.plot(daily.index, residual, color="tab:blue", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(title="Остатки лучшей модели", xlabel="Дата UTC", ylabel="Наблюдение − модель")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "regression_residuals.png", dpi=180)
    if show_plots:
        plt.show()
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    axes[0].plot(raw.index, raw, color="tab:green", linewidth=0.45)
    axes[0].set(title="Исходные измерения NMDB", ylabel="RCORR_E")
    axes[1].plot(daily.index, daily, color="tab:purple", linewidth=0.8)
    axes[1].set(title="После агрегации до суток", ylabel="RCORR_E", xlabel="Дата UTC")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "raw_and_daily.png", dpi=180)
    if show_plots:
        plt.show()
    plt.close(fig)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(comparison.to_string(index=False))
    return result


if __name__ == "__main__":
    main()
