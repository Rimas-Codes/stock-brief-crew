from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

CSV = Path('data/predictions.csv')
Z80 = 1.2816  # 80% two-sided interval


def ewma_var(r, lam=0.94):
    """v[t] = variance forecast for r[t] using data before t; v[-1] is for tomorrow."""
    v = np.empty(len(r) + 1)
    v[0] = r.iloc[:30].var()
    for t in range(len(r)):
        v[t + 1] = lam * v[t] + (1 - lam) * r.iloc[t] ** 2
    return v


def forecast_next_close(ticker: str, drift: float = 0.0) -> dict:
    close = yf.Ticker(ticker).history(period='3y', auto_adjust=True)['Close'].dropna()
    if len(close) < 250:
        raise ValueError(f'Not enough price history for {ticker}')
    r = np.log(close).diff().dropna()
    sigma = float(np.sqrt(ewma_var(r)[-1]))
    last = float(close.iloc[-1])
    return {
        'ticker': ticker,
        'asof': close.index[-1].date().isoformat(),
        'last_close': round(last, 2),
        'pred_close': round(last * np.exp(drift), 2),
        'pred_low': round(last * np.exp(drift - Z80 * sigma), 2),
        'pred_high': round(last * np.exp(drift + Z80 * sigma), 2),
        'sigma_1d_pct': round(sigma * 100, 2),
        'method': 'ewma_vol_v1',
    }


def log_prediction(fc: dict):
    CSV.parent.mkdir(exist_ok=True)
    if CSV.exists():
        df = pd.read_csv(CSV)
        if ((df['ticker'] == fc['ticker']) & (df['asof'] == fc['asof'])).any():
            return   # already logged this forecast

    row = {**fc, 'run_date': date.today().isoformat(), 'target_date': None,
           'actual_close': None, 'abs_error': None,
           'naive_abs_error': None, 'in_interval': None}
    pd.DataFrame([row]).to_csv(CSV, mode='a', header=not CSV.exists(), index=False)


def fill_actuals(ticker: str):
    """Score earlier forecasts once the next trading day's close exists."""
    if not CSV.exists():
        return
    df = pd.read_csv(CSV)
    df['target_date'] = df['target_date'].astype('object')
    hist = yf.Ticker(ticker).history(period='1mo', auto_adjust=True)['Close']
    hist.index = hist.index.strftime('%Y-%m-%d')
    todo = df[(df['ticker'] == ticker) & df['actual_close'].isna()].index
    for i in todo:
        later = hist[hist.index > df.at[i, 'asof']]
        if later.empty:
            continue
        actual = float(later.iloc[0])
        df.at[i, 'target_date'] = later.index[0]
        df.at[i, 'actual_close'] = actual
        df.at[i, 'abs_error'] = abs(actual - df.at[i, 'pred_close'])
        df.at[i, 'naive_abs_error'] = abs(actual - df.at[i, 'last_close'])
        df.at[i, 'in_interval'] = int(df.at[i, 'pred_low'] <= actual <= df.at[i, 'pred_high'])
    df.to_csv(CSV, index=False)


def track_record(ticker: str) -> str:
    if not CSV.exists():
        return 'No scored forecasts yet.'
    df = pd.read_csv(CSV)
    s = df[(df['ticker'] == ticker) & df['actual_close'].notna()]
    if s.empty:
        return 'No scored forecasts yet.'
    return (f'{len(s)} scored forecasts: 80% range contained the actual close '
            f'{s["in_interval"].mean():.0%} of the time; mean absolute error '
            f'${s["abs_error"].mean():.2f} vs ${s["naive_abs_error"].mean():.2f} '
            f'for "tomorrow equals today".')