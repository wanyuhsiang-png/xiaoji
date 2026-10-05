"""
上漲時的加減碼規則回測：以「MA200 減碼一半」為基準，比較漲多減碼、移動停利、突破加碼。

部位為 0~1（1 = 滿倉），訊號以當日收盤判斷、隔日起承擔報酬。

用法：
  python upside.py --ticker 2330.TW --stock
"""
import argparse

import numpy as np
import pandas as pd

from backtest import FEE, TAX_ETF, TAX_STOCK, _has_tabulate, run, stats


def load_ohlc(ticker, start):
    import yfinance as yf

    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


def atr(df, n=14):
    c, h, l = df["Close"], df["High"], df["Low"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def base(c):
    """基準：MA200 之上滿倉，跌破減碼一半，站回買回。"""
    return pd.Series(np.where(c > c.rolling(200).mean(), 1.0, 0.5), index=c.index)


def bias_trim(c, high, low, cut=1 / 3):
    """漲多減碼：股價高於 MA200 超過 high 時減碼 cut，乖離回落到 low 以下再買回。"""
    ma = c.rolling(200).mean()
    bias = c / ma - 1
    pos, trimmed = [], False
    for b, x, m in zip(bias, c, ma):
        if np.isnan(m):
            pos.append(0.5)
            continue
        if not trimmed and b > high:
            trimmed = True
        elif trimmed and b < low:
            trimmed = False
        full = 1.0 if x > m else 0.5
        pos.append(full * (1 - cut) if trimmed else full)
    return pd.Series(pos, index=c.index)


def trailing(df, k=3.0, lookback=60):
    """移動停利：收盤從近 lookback 日最高收盤回落超過 k 倍 ATR 時減碼一半，創 20 日新高再買回。"""
    c = df["Close"]
    ma = c.rolling(200).mean()
    peak, a, hi20 = c.rolling(lookback).max(), atr(df), c.rolling(20).max()
    pos, stopped = [], False
    for x, m, p, v, h in zip(c, ma, peak, a, hi20):
        if np.isnan(m) or np.isnan(v):
            pos.append(0.5)
            continue
        if not stopped and x < p - k * v:
            stopped = True
        elif stopped and x >= h:
            stopped = False
        full = 1.0 if x > m else 0.5
        pos.append(full * 0.5 if stopped else full)
    return pd.Series(pos, index=c.index)


def breakout_add(c, normal=0.7, lookback=60):
    """突破加碼：平時持有 normal，收盤創 lookback 日新高加到滿倉，跌破月線退回 normal。"""
    ma200, ma20 = c.rolling(200).mean(), c.rolling(20).mean()
    prior_high = c.shift(1).rolling(lookback).max()
    pos, added = [], False
    for x, m, m20, ph in zip(c, ma200, ma20, prior_high):
        if np.isnan(m):
            pos.append(normal * 0.5)
            continue
        if not added and x > ph:
            added = True
        elif added and x < m20:
            added = False
        if x <= m:
            added = False
        full = 1.0 if added else normal
        pos.append(full if x > m else normal * 0.5)
    return pd.Series(pos, index=c.index)


def strategies(df):
    c = df["Close"]
    return [
        ("基準：MA200 減碼一半", base(c)),
        ("漲多減碼：乖離>30% 減 1/3", bias_trim(c, 0.30, 0.15)),
        ("漲多減碼：乖離>40% 減 1/3", bias_trim(c, 0.40, 0.20)),
        ("移動停利：回落 3ATR 減半", trailing(df, 3.0)),
        ("移動停利：回落 5ATR 減半", trailing(df, 5.0)),
        ("突破加碼：平時 7 成，創高滿倉", breakout_add(c, 0.7)),
        ("買進持有（參考）", pd.Series(1.0, index=c.index)),
    ]


def report(df, mask, buy_cost, sell_cost, title):
    sub = df["Close"][mask]
    rows = []
    for name, pos in strategies(df):
        s = stats(name, *run(sub, pos[mask], buy_cost, sell_cost))
        rows.append({k: s[k] for k in ("策略", "年化報酬", "最大回撤", "報酬/回撤", "持倉時間", "交易次數")})
    out = pd.DataFrame(rows).set_index("策略").rename(columns={"持倉時間": "平均持股"})
    print(f"\n### {title}（{sub.index[0]:%Y-%m-%d} ~ {sub.index[-1]:%Y-%m-%d}）\n")
    print(out.to_markdown() if _has_tabulate() else out.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="0050.TW")
    ap.add_argument("--start", default="2014-01-03")
    ap.add_argument("--stock", action="store_true")
    ap.add_argument("--fee-discount", type=float, default=0.28)
    ap.add_argument("--split", default="2019-01-01")
    args = ap.parse_args()

    df = load_ohlc(args.ticker, args.start)
    jumps = df["Close"].pct_change().abs()
    if (jumps > 0.11).any():
        print(f"⚠️ 資料異常日：{', '.join(f'{d:%Y-%m-%d}' for d in jumps[jumps > 0.11].index)}")
    fee = FEE * args.fee_discount
    bc, sc = fee, fee + (TAX_STOCK if args.stock else TAX_ETF)
    print(f"標的：{args.ticker}")
    # 前 200 日為均線暖身期，不納入比較
    idx = df.index
    warm = idx >= idx[min(200, len(idx) - 1)]
    split = pd.Timestamp(args.split)
    for title, m in [("全期間", warm), ("樣本內", warm & (idx < split)), ("樣本外", idx >= split)]:
        if m.sum() > 250:
            report(df, m, bc, sc, title)


if __name__ == "__main__":
    main()
