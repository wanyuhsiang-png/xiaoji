"""
個股健檢報告：趨勢、技術指標、法人籌碼、新聞，並給出規則化的操作建議與參考價位。

主要依據為回測驗證過的趨勢規則（MA200 減碼一半）；KD、布林、法人、新聞僅作參考，
回測顯示它們對短期漲跌的預測力很弱。

用法：
  python analyze.py 2330
  python analyze.py 2330 0050 2454
  環境變數 FINMIND_TOKEN 可提高 FinMind API 額度（非必要）
"""
import argparse
import os
import sys
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from email.utils import parsedate_to_datetime

import numpy as np
import pandas as pd
import requests

from backtest import FEE, TAX_ETF, TAX_STOCK, run, signal_buy_hold, signal_ma200_half, stats

FINMIND = "https://api.finmindtrade.com/api/v4/data"


# ---------- 資料 ----------

def finmind(dataset, data_id, start):
    headers = {}
    if os.environ.get("FINMIND_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['FINMIND_TOKEN']}"
    r = requests.get(FINMIND, params={"dataset": dataset, "data_id": data_id, "start_date": start},
                     headers=headers, timeout=60)
    r.raise_for_status()
    return pd.DataFrame(r.json().get("data", []))


def load_ohlc(code):
    import yfinance as yf

    for suffix in (".TW", ".TWO"):
        df = yf.download(code + suffix, start="2014-01-03", auto_adjust=True, progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return code + suffix, df.dropna()
    raise SystemExit(f"找不到 {code} 的股價")


def load_name(code):
    try:
        df = finmind("TaiwanStockInfo", code, "2000-01-01")
        return df["stock_name"].iloc[0], df["industry_category"].iloc[0]
    except Exception:
        return "", ""


def load_institutional(code):
    """近 60 日法人買賣超（張）。"""
    start = (date.today() - timedelta(days=90)).isoformat()
    df = finmind("TaiwanStockInstitutionalInvestorsBuySell", code, start)
    if df.empty:
        return None
    df["net"] = (df["buy"] - df["sell"]) / 1000
    group = {"Foreign_Investor": "外資", "Foreign_Dealer_Self": "外資",
             "Investment_Trust": "投信", "Dealer_self": "自營商", "Dealer_Hedging": "自營商"}
    df["who"] = df["name"].map(group)
    return df.dropna(subset=["who"]).pivot_table(index="date", columns="who", values="net", aggfunc="sum").fillna(0)


def load_margin(code):
    start = (date.today() - timedelta(days=60)).isoformat()
    df = finmind("TaiwanStockMarginPurchaseShortSale", code, start)
    if df.empty:
        return None
    return df.set_index("date")[["MarginPurchaseTodayBalance", "ShortSaleTodayBalance"]]


def load_news(query, days=10, limit=8):
    url = "https://news.google.com/rss/search"
    params = {"q": f"{query} when:{days}d", "hl": "zh-TW", "gl": "TW", "ceid": "TW:zh-Hant"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    items = []
    for it in ET.fromstring(r.content).iter("item"):
        pub = it.findtext("pubDate")
        items.append({
            "ts": parsedate_to_datetime(pub) if pub else None,
            "title": it.findtext("title", ""),
            "link": it.findtext("link", ""),
        })
    items = sorted((i for i in items if i["ts"]), key=lambda i: i["ts"], reverse=True)[:limit]
    for i in items:
        i["date"] = i["ts"].strftime("%m-%d")
    return items


# ---------- 指標 ----------

def indicators(df):
    c, h, l = df["Close"], df["High"], df["Low"]
    out = pd.DataFrame(index=df.index)
    out["close"] = c
    for n in (5, 20, 60, 120, 200, 240):
        out[f"ma{n}"] = c.rolling(n).mean()
    # KD(9,3,3)
    low9, high9 = l.rolling(9).min(), h.rolling(9).max()
    rsv = ((c - low9) / (high9 - low9) * 100).fillna(50)
    k, d, ks, ds = 50.0, 50.0, [], []
    for v in rsv:
        k = k * 2 / 3 + v / 3
        d = d * 2 / 3 + k / 3
        ks.append(k)
        ds.append(d)
    out["k"], out["d"] = ks, ds
    # 布林通道(20, 2)
    std20 = c.rolling(20).std()
    out["bb_up"], out["bb_mid"], out["bb_low"] = out["ma20"] + 2 * std20, out["ma20"], out["ma20"] - 2 * std20
    # RSI(14)、MACD(12,26,9)、ATR(14)
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    out["rsi"] = 100 - 100 / (1 + up / dn)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    out["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    out["atr"] = tr.rolling(14).mean()
    out["vol"] = df["Volume"] / 1000
    out["vol_ma20"] = out["vol"].rolling(20).mean()
    out["high20"], out["low20"] = h.rolling(20).max(), l.rolling(20).min()
    out["high60"] = h.rolling(60).max()
    out["high240"] = h.rolling(240).max()
    return out


def tick(price):
    """台股升降單位。"""
    for limit, step in ((10, 0.01), (50, 0.05), (100, 0.1), (500, 0.5), (1000, 1)):
        if price < limit:
            return step
    return 5


def rnd(price):
    t = tick(price)
    return round(round(price / t) * t, 2)


# ---------- 判斷 ----------

def assess(ind):
    x, prev5 = ind.iloc[-1], ind.iloc[-6]
    c = x.close
    notes, score = [], 0

    above200 = c > x.ma200
    ma60_up = x.ma60 > prev5.ma60
    bull_align = x.ma20 > x.ma60 > x.ma200
    bear_align = x.ma20 < x.ma60 < x.ma200

    # 主要依據：趨勢
    if above200 and ma60_up:
        trend, action = "多頭", "續抱／可分批布局"
    elif above200:
        trend, action = "多頭但走緩", "續抱，不追高"
    else:
        trend, action = "空頭", "依規則減碼一半，不新增部位"

    # 參考：短線位置
    hot = x.k > 80 and c > x.bb_up * 0.99
    cold = x.k < 20 and c < x.bb_low * 1.01
    if above200 and hot:
        timing = "短線過熱（KD 高檔、貼近布林上軌），不追價，等回檔到買點區"
    elif above200 and cold:
        timing = "多頭中的回檔（KD 低檔、貼近布林下軌），是分批買點"
    elif not above200 and cold:
        timing = "空頭中的超跌，可能有反彈，但不建議搶"
    elif x.k > x.d and ind["k"].iloc[-2] <= ind["d"].iloc[-2]:
        timing = "KD 剛黃金交叉，短線轉強"
    elif x.k < x.d and ind["k"].iloc[-2] >= ind["d"].iloc[-2]:
        timing = "KD 剛死亡交叉，短線轉弱"
    else:
        timing = "短線無明顯訊號"

    # 價位
    supports = sorted({rnd(v) for v in (x.ma20, x.ma60, x.bb_low, x.low20) if v < c}, reverse=True)
    resists = sorted({rnd(v) for v in (x.bb_up, x.high60, x.high240) if v > c * 1.005})
    buy1 = supports[0] if supports else rnd(c)
    buy2 = supports[1] if len(supports) > 1 else rnd(buy1 - x.atr)
    stop = rnd(min(buy2, x.ma60 if above200 else buy2) - x.atr)
    levels = {
        "現價": rnd(c),
        "第一買點": buy1,
        "第二買點": buy2,
        "停損": stop,
        "減碼線（MA200）": rnd(x.ma200),
        "壓力": resists[:2],
    }
    flags = {
        "above200": above200, "ma60_up": ma60_up, "bull_align": bull_align, "bear_align": bear_align,
    }
    return trend, action, timing, levels, flags


# ---------- 報告 ----------

def fmt(v, pct=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{v:+.1%}" if pct else f"{v:,.2f}"


def report(code, is_stock):
    ticker, df = load_ohlc(code)
    name, industry = load_name(code)
    ind = indicators(df)
    x = ind.iloc[-1]
    trend, action, timing, lv, fl = assess(ind)

    out = []
    p = out.append
    p(f"## {code} {name}（{industry or ticker}）｜{ind.index[-1]:%Y-%m-%d} 收盤 {fmt(x.close)}\n")
    p(f"### 結論：**{trend}** → {action}\n")
    p(f"- 短線位置：{timing}")
    p(f"- 判斷依據：收盤{'站上' if fl['above200'] else '跌破'} MA200；MA60 {'上揚' if fl['ma60_up'] else '下彎'}"
      f"{'；均線多頭排列' if fl['bull_align'] else '；均線空頭排列' if fl['bear_align'] else ''}\n")

    p("### 參考價位\n")
    p("| 項目 | 價位 | 說明 |\n|---|---|---|")
    p(f"| 第一買點 | {lv['第一買點']} | 最近支撐（月線／布林下軌／20 日低擇近） |")
    p(f"| 第二買點 | {lv['第二買點']} | 下一層支撐 |")
    p(f"| 停損 | {lv['停損']} | 第二買點再扣 1 倍 ATR，跌破代表支撐失效 |")
    p(f"| 減碼線 | {lv['減碼線（MA200）']} | 收盤跌破 MA200 減碼一半（主要規則） |")
    for i, r in enumerate(lv["壓力"], 1):
        p(f"| 壓力 {i} | {r} | 布林上軌／60 日高／52 週高 |")
    risk = lv["第一買點"] - lv["停損"]
    if risk > 0:
        shares = 10000 / risk
        qty = f"{shares / 1000:.1f} 張" if shares >= 1000 else f"{shares:,.0f} 股（零股）"
        p(f"\n部位參考：資金 100 萬、單筆最多虧 1%（1 萬元）→ 在第一買點最多買約 **{qty}**"
          f"，約 {shares * lv['第一買點'] / 1e4:,.0f} 萬元\n")

    p("### 技術指標\n")
    p("| 指標 | 數值 | 狀態 |\n|---|---|---|")
    for n, label in ((5, "週線"), (20, "月線"), (60, "季線"), (120, "半年線"), (240, "年線")):
        v = x[f"ma{n}"]
        p(f"| MA{n} {label} | {fmt(v)} | 股價{'在上' if x.close > v else '在下'}（{fmt(x.close / v - 1, True)}） |")
    kd_state = "高檔區" if x.k > 80 else "低檔區" if x.k < 20 else "中性"
    p(f"| KD(9) | K {x.k:.0f} / D {x.d:.0f} | {kd_state}，{'K > D' if x.k > x.d else 'K < D'} |")
    bw = (x.bb_up - x.bb_low) / x.bb_mid
    pos = (x.close - x.bb_low) / (x.bb_up - x.bb_low)
    p(f"| 布林通道 | {fmt(x.bb_low)} ~ {fmt(x.bb_up)} | 位於通道 {pos:.0%} 位置，寬度 {bw:.1%} |")
    p(f"| RSI(14) | {x.rsi:.0f} | {'過熱' if x.rsi > 70 else '超賣' if x.rsi < 30 else '中性'} |")
    p(f"| MACD 柱 | {x.macd_hist:+.2f} | {'多方' if x.macd_hist > 0 else '空方'}"
      f"{'，柱狀放大' if abs(x.macd_hist) > abs(ind['macd_hist'].iloc[-2]) else '，柱狀收斂'} |")
    p(f"| ATR(14) | {fmt(x.atr)} | 日均波動 {x.atr / x.close:.1%} |")
    p(f"| 成交量 | {x.vol:,.0f} 張 | 為 20 日均量 {x.vol / x.vol_ma20:.1f} 倍 |\n")

    p("### 法人與資券（參考）\n")
    try:
        inst = load_institutional(code)
        if inst is not None and not inst.empty:
            p("| 法人 | 今日 | 近 5 日 | 近 20 日 |\n|---|---|---|---|")
            for who in ("外資", "投信", "自營商"):
                if who in inst:
                    s = inst[who]
                    p(f"| {who} | {s.iloc[-1]:+,.0f} 張 | {s.tail(5).sum():+,.0f} 張 | {s.tail(20).sum():+,.0f} 張 |")
            p(f"\n（資料日 {inst.index[-1]}）")
    except Exception as e:
        p(f"法人資料取得失敗：{e}")
    try:
        m = load_margin(code)
        if m is not None and len(m) > 5:
            mp, ss = m["MarginPurchaseTodayBalance"], m["ShortSaleTodayBalance"]
            p(f"\n融資餘額 {mp.iloc[-1]:,.0f} 張（5 日 {mp.iloc[-1] - mp.iloc[-6]:+,.0f}）；"
              f"融券餘額 {ss.iloc[-1]:,.0f} 張（5 日 {ss.iloc[-1] - ss.iloc[-6]:+,.0f}）")
    except Exception as e:
        p(f"資券資料取得失敗：{e}")

    p("\n### 近期新聞\n")
    try:
        news = load_news(f"{code} {name}".strip())
        if news:
            for n in news:
                p(f"- {n['date']} [{n['title']}]({n['link']})")
        else:
            p("近 10 天無相關新聞")
    except Exception as e:
        p(f"新聞取得失敗：{e}")

    # 本檔歷史回測，驗證主要規則
    fee = FEE * 0.28
    bc, sc = fee, fee + (TAX_STOCK if is_stock else TAX_ETF)
    c = df["Close"]
    rows = [stats(n, *run(c, fn(c), bc, sc)) for n, fn in (("買進持有", signal_buy_hold), ("MA200 減碼一半", signal_ma200_half))]
    p(f"\n### 主要規則在本檔的歷史表現（{c.index[0]:%Y} ~ {c.index[-1]:%Y}）\n")
    p("| 策略 | 年化報酬 | 最大回撤 |\n|---|---|---|")
    for r in rows:
        p(f"| {r['策略']} | {r['年化報酬']} | {r['最大回撤']} |")

    p("\n> 本報告為規則化整理，非投資建議。KD、布林、法人、新聞經回測對短期漲跌預測力有限，請以趨勢規則與停損紀律為主。\n")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("codes", nargs="+", help="股票代號，例如 2330 0050")
    args = ap.parse_args()
    for code in args.codes:
        code = code.strip().upper()
        try:
            # 代號 00 開頭視為 ETF（證交稅 0.1%）
            print(report(code, is_stock=not code.startswith("00")))
        except SystemExit as e:
            print(f"## {code}\n\n{e}\n")
        except Exception as e:
            print(f"## {code}\n\n分析失敗：{e}\n", file=sys.stdout)


if __name__ == "__main__":
    main()
