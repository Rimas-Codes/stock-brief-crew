import sys

import numpy as np
import pandas as pd
import yfinance as yf

from stck_analyzer.forecast import ewma_var, Z80
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # .../stck_analyzer (the folder with pyproject.toml)

def backfill(ticker: str, days: int = 250, out=None):
    out = Path(out) if out else ROOT / 'data' / 'backtest.csv'
    close = yf.Ticker(ticker).history(period='3y', auto_adjust=True)['Close'].dropna()
    r = np.log(close).diff().dropna()
    v = ewma_var(r)   # v[t] = variance forecast made at close.index[t] for close.index[t+1]

    rows = []
    for t in range(len(r) - days, len(r)):
        last = float(close.iloc[t])
        actual = float(close.iloc[t + 1])
        s = float(np.sqrt(v[t]))
        lo = round(last * np.exp(-Z80 * s), 2)
        hi = round(last * np.exp(Z80 * s), 2)
        asof = close.index[t].date().isoformat()
        rows.append({
            'ticker': ticker, 'asof': asof, 'last_close': round(last, 2),
            'pred_close': round(last, 2), 'pred_low': lo, 'pred_high': hi,
            'sigma_1d_pct': round(s * 100, 2), 'method': 'ewma_vol_v1',
            'run_date': asof, 'target_date': close.index[t + 1].date().isoformat(),
            'actual_close': round(actual, 2),
            'abs_error': round(abs(actual - last), 2),
            'naive_abs_error': round(abs(actual - last), 2),
            'in_interval': int(lo <= actual <= hi),
        })
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'Wrote {len(rows)} rows, coverage {np.mean([x["in_interval"] for x in rows]):.1%}')
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == '__main__':
    backfill(sys.argv[1] if len(sys.argv) > 1 else 'NVDA')