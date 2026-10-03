"""
外資買賣超訊號回測

1. 預測力檢驗：依外資買賣超分五組，看之後 1／5／20 日平均報酬
2. 策略回測：外資訊號進出場，與買進持有、MA200 比較

資料：FinMind 三大法人（大盤或個股），股價用 yfinance。
外資買賣超於收盤後公布，訊號以當日資料判斷、隔日起承擔報酬。

用法：
  python foreign_flow.py                         # 大盤外資 × 0050
  python foreign_flow.py --stock-id 2330 --stock # 個股外資 × 2330
  環境變數 FINMIND_TOKEN 可提高 API 額度（非必要）
"""
import argparse
import os

import numpy as np
import pandas as pd
import requests

from backtest import FEE, TAX_ETF, TAX_STOCK, _has_tabulate, run, signal_buy_hold, signal_ma200, stats

FINMIND = "https://api.finmindtrade.com/api/v4/data"


def finmind(dataset, start, data_id=None):
    params = {"dataset": dataset, "start_date": start}
    if data_id:
        params["data_id"] = data_id
    headers = {}
    if os.environ.get("FINMIND_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['FINMIND_TOKEN']}"
    r = requests.get(FINMIND, params=params, headers=headers, timeout=60)
    r.raise_for_status()
    js = r.json()
    if not js.get("data"):
        raise SystemExit(f"FinMind 無資料：{js.get('msg')}")
    return pd.DataFrame(js["data"])


def load_foreign(start, stock_id=None):
    """回傳每日外資（不含自營）買賣超金額或股數。"""
    if stock_id:
        df = finmind("TaiwanStockInstitutionalInvestorsBuySell", start, stock_id)
    else:
        df = finmind("TaiwanStockTotalInstitutionalInvestors", start)
    names = sorted(df["name"].unique())
    print(f"法人類別：{names}")
    foreign = [n for n in names if n == "Foreign_Investor"] or [n for n in names if "Foreign" in n and "Dealer" not in n]
    if not foreign:
        raise SystemExit("找不到外資欄位")
    df = df[df["name"].isin(foreign)]
    net = (df["buy"].astype(float) - df["sell"].astype(float)).groupby(pd.to_datetime(df["date"])).sum()
    return net.sort_index()


def load_close(ticker, start):
    import yfinance as yf

    close = yf.download(ticker, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.dropna()


def print_df(df):
    print(df.to_markdown() if _has_tabulate() else df.to_string())


def predictive_test(close, net):
    """依訊號分五組，計算之後 N 日報酬（隔日起算，無前視）。"""
    signals = {
        "當日買賣超": net,
        "5 日累積買賣超": net.rolling(5).sum(),
        "20 日累積買賣超": net.rolling(20).sum(),
    }
    for name, sig in signals.items():
        rows = {}
        for h in (1, 5, 20):
            fwd = close.shift(-h) / close - 1
            df = pd.DataFrame({"sig": sig, "fwd": fwd}).dropna()
            df["組別"] = pd.qcut(df["sig"], 5, labels=["1 大賣", "2", "3", "4", "5 大買"])
            g = df.groupby("組別", observed=True)["fwd"]
            rows[f"{h} 日後報酬"] = g.mean().map("{:+.2%}".format)
            rows[f"{h} 日上漲機率"] = g.apply(lambda x: (x > 0).mean()).map("{:.0%}".format)
        base = {f"{h} 日": f"{(close.shift(-h) / close - 1).mean():+.2%}" for h in (1, 5, 20)}
        print(f"\n#### {name}（全樣本平均：{base}）\n")
        print_df(pd.DataFrame(rows))


def signal_streak(net, n=3):
    """連續 n 日買超進場，連續 n 日賣超出場。"""
    buy = (net > 0).astype(int).rolling(n).sum() == n
    sell = (net < 0).astype(int).rolling(n).sum() == n
    pos, holding = [], 0
    for b, s in zip(buy, sell):
        if b:
            holding = 1
        elif s:
            holding = 0
        pos.append(holding)
    return pd.Series(pos, index=net.index)


def strategies(close, net):
    sum20 = net.rolling(20).sum()
    ma200 = signal_ma200(close)
    return [
        ("買進持有", signal_buy_hold(close)),
        ("MA200 趨勢", ma200),
        ("外資連 3 買進／連 3 賣出", signal_streak(net, 3)),
        ("外資 20 日累積買超", (sum20 > 0).astype(int)),
        ("外資 20 日累積賣超時買（反向）", (sum20 < 0).astype(int)),
        ("MA200 且外資 20 日買超", ((ma200 == 1) & (sum20 > 0)).astype(int)),
        ("MA200 或外資 20 日買超", ((ma200 == 1) | (sum20 > 0)).astype(int)),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="0050.TW")
    ap.add_argument("--stock-id", help="個股代號（使用個股外資買賣超），預設用大盤外資")
    ap.add_argument("--start", default="2014-01-03")
    ap.add_argument("--stock", action="store_true", help="個股證交稅 0.3%")
    ap.add_argument("--fee-discount", type=float, default=0.28)
    args = ap.parse_args()
    ticker = f"{args.stock_id}.TW" if args.stock_id else args.ticker

    net = load_foreign(args.start, args.stock_id)
    close = load_close(ticker, args.start)
    # 對齊交易日；先用全部股價算均線，再裁到有外資資料的期間
    net = net.reindex(close.index).fillna(0)
    first = net.ne(0).idxmax()
    unit = "股" if args.stock_id else "元"
    print(f"標的：{ticker}｜外資資料：{'個股' if args.stock_id else '大盤'}（{unit}）｜{first:%Y-%m-%d} ~ {close.index[-1]:%Y-%m-%d}")

    print("\n### 一、預測力檢驗：外資買賣超分五組後的平均報酬")
    mask = close.index >= first
    predictive_test(close[mask], net[mask])

    fee = FEE * args.fee_discount
    buy_cost, sell_cost = fee, fee + (TAX_STOCK if args.stock else TAX_ETF)
    # 均線暖身：從外資資料起始後 200 日開始比較，各策略同一起點
    start = close.index[close.index.get_loc(first) + 200]
    rows = []
    for name, pos in strategies(close, net):
        m = close.index >= start
        rows.append(stats(name, *run(close[m], pos[m], buy_cost, sell_cost)))
    print(f"\n### 二、策略回測（{start:%Y-%m-%d} ~ {close.index[-1]:%Y-%m-%d}）\n")
    print_df(pd.DataFrame(rows).set_index("策略"))


if __name__ == "__main__":
    main()
