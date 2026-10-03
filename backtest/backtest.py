"""
台股策略回測：比較三種做法的期望值
  1. 買進持有（Buy & Hold）
  2. 200 日均線趨勢（收盤 > MA200 持有，否則空手）
  3. RSI(2) 均值回歸（MA200 之上且 RSI2 < 10 買進，收盤 > MA5 賣出）

訊號以當日收盤判斷、隔日起承擔報酬（避免前視偏差），並扣除手續費與證交稅。

用法：
  pip install pandas numpy yfinance
  python backtest.py                      # 預設 0050.TW
  python backtest.py --ticker 006208.TW
  python backtest.py --csv my_data.csv    # 自備資料，需有 Date、Close 欄位（請用還原權息價）
  python backtest.py --demo               # 模擬資料，僅供驗證程式
"""
import argparse

import numpy as np
import pandas as pd

FEE = 0.001425  # 券商手續費（單邊），可用 --fee-discount 打折
TAX_ETF = 0.001  # ETF 證交稅（賣出時）
TAX_STOCK = 0.003  # 個股證交稅（賣出時）


def load_prices(args):
    if args.demo:
        rng = np.random.default_rng(42)
        n = 252 * 17
        r = rng.normal(0.0004, 0.012, n)
        dates = pd.bdate_range("2008-01-01", periods=n)
        return pd.Series(100 * np.cumprod(1 + r), index=dates, name="Close")
    if args.csv:
        df = pd.read_csv(args.csv, parse_dates=["Date"]).set_index("Date").sort_index()
        return df["Close"].astype(float).dropna()
    import yfinance as yf

    df = yf.download(args.ticker, start=args.start, auto_adjust=True, progress=False)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.dropna()


def rsi(close, n):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def signal_buy_hold(close):
    return pd.Series(1, index=close.index)


def signal_ma200(close):
    return (close > close.rolling(200).mean()).astype(int)


def signal_ma200_half(close):
    """MA200 之上全額持有，跌破只減碼到一半。"""
    return pd.Series(np.where(close > close.rolling(200).mean(), 1.0, 0.5), index=close.index)


def signal_rsi2(close):
    ma200 = close.rolling(200).mean()
    ma5 = close.rolling(5).mean()
    r2 = rsi(close, 2)
    pos, holding = [], 0
    for c, m200, m5, r in zip(close, ma200, ma5, r2):
        if holding and c > m5:
            holding = 0
        elif not holding and not np.isnan(m200) and c > m200 and r < 10:
            holding = 1
        pos.append(holding)
    return pd.Series(pos, index=close.index)


def run(close, pos, buy_cost, sell_cost):
    """pos[t] 為第 t 日收盤後的持倉（0~1），承擔第 t+1 日報酬。
    逐筆交易以「加碼到賣出」計算；半倉策略即為可調整的那一半部位。"""
    ret = close.pct_change().fillna(0)
    held = pos.shift(1).fillna(0)
    change = pos.diff().fillna(pos.iloc[0])
    # 成本依部位變動比例計算（支援半倉）
    cost = change.clip(lower=0) * buy_cost + (-change).clip(lower=0) * sell_cost
    daily = held * ret - cost
    equity = (1 + daily).cumprod()

    # 逐筆交易
    trades, entry = [], None
    for i, (p, ch) in enumerate(zip(pos, change)):
        if ch > 0:
            entry = i
        elif ch < 0 and entry is not None:
            gross = close.iloc[i] / close.iloc[entry]
            trades.append(gross * (1 - sell_cost) / (1 + buy_cost) - 1)
            entry = None
    if entry is not None:  # 期末未平倉，以最後收盤計（假設賣出）
        gross = close.iloc[-1] / close.iloc[entry]
        trades.append(gross * (1 - sell_cost) / (1 + buy_cost) - 1)
    return equity, np.array(trades), held.mean()


def stats(name, equity, trades, exposure):
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = equity.iloc[-1] ** (1 / years) - 1
    mdd = (equity / equity.cummax() - 1).min()
    wins, losses = trades[trades > 0], trades[trades <= 0]
    win_rate = len(wins) / len(trades) if len(trades) else np.nan
    avg_w = wins.mean() if len(wins) else 0
    avg_l = losses.mean() if len(losses) else 0
    pf = wins.sum() / -losses.sum() if losses.sum() < 0 else np.inf
    return {
        "策略": name,
        "年化報酬": f"{cagr:.1%}",
        "最大回撤": f"{mdd:.1%}",
        "報酬/回撤": f"{cagr / -mdd:.2f}" if mdd < 0 else "-",
        "持倉時間": f"{exposure:.0%}",
        "交易次數": len(trades),
        "勝率": f"{win_rate:.0%}",
        "平均賺": f"{avg_w:+.2%}",
        "平均賠": f"{avg_l:+.2%}",
        "每筆期望值": f"{trades.mean():+.2%}" if len(trades) else "-",
        "獲利因子": f"{pf:.2f}",
    }


STRATEGIES = [
    ("買進持有", signal_buy_hold),
    ("MA200 趨勢", signal_ma200),
    ("MA200 減碼一半", signal_ma200_half),
    ("RSI(2) 均值回歸", signal_rsi2),
]


def report(close, mask, buy_cost, sell_cost, title):
    # 訊號用全期間資料計算再切片，避免切分處均線暖身期失真
    sub = close[mask]
    rows = [stats(name, *run(sub, fn(close)[mask], buy_cost, sell_cost)) for name, fn in STRATEGIES]
    df = pd.DataFrame(rows).set_index("策略")
    print(f"\n### {title}（{sub.index[0]:%Y-%m-%d} ~ {sub.index[-1]:%Y-%m-%d}）\n")
    print(df.to_markdown() if _has_tabulate() else df.to_string())


def _has_tabulate():
    try:
        import tabulate  # noqa: F401

        return True
    except ImportError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="0050.TW")
    ap.add_argument("--start", default="2008-01-01")
    ap.add_argument("--csv")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--stock", action="store_true", help="個股（證交稅 0.3%），預設為 ETF（0.1%）")
    ap.add_argument("--fee-discount", type=float, default=1.0, help="手續費折數，例如 0.28 代表 2.8 折")
    ap.add_argument("--split", default="2019-01-01", help="樣本內/樣本外切分日")
    args = ap.parse_args()

    close = load_prices(args)
    if close.empty:
        raise SystemExit("抓不到資料，請確認代號或網路")
    # 台股單日漲跌幅上限 10%，超過代表資料有誤（常見於分割／還原權息處理錯誤）
    jumps = close.pct_change().abs()
    bad = jumps[jumps > 0.11]
    if not bad.empty:
        print("⚠️ 資料異常：以下日期單日漲跌超過 11%，結果可能失真")
        for d, v in bad.items():
            print(f"  {d:%Y-%m-%d}  {close.pct_change()[d]:+.1%}")
    fee = FEE * args.fee_discount
    buy_cost, sell_cost = fee, fee + (TAX_STOCK if args.stock else TAX_ETF)
    label = "模擬資料" if args.demo else (args.csv or args.ticker)
    print(f"標的：{label}｜買進成本 {buy_cost:.4%}｜賣出成本 {sell_cost:.4%}")

    split = pd.Timestamp(args.split)
    for title, mask in [
        ("全期間", np.ones(len(close), dtype=bool)),
        ("樣本內", close.index < split),
        ("樣本外", close.index >= split),
    ]:
        if mask.sum() > 250:
            report(close, mask, buy_cost, sell_cost, title)

if __name__ == "__main__":
    main()
