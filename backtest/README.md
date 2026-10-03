# 台股策略期望值回測

比較三種做法在台股 ETF 的期望值、勝率與風險：

| 策略 | 規則 |
|---|---|
| 買進持有 | 一直持有 |
| MA200 趨勢 | 收盤 > 200 日均線持有，否則空手 |
| RSI(2) 均值回歸 | 收盤 > MA200 且 RSI(2) < 10 買進；收盤 > MA5 賣出 |

- 訊號以當日收盤判斷、隔日才承擔報酬（無前視偏差）
- 扣除手續費 0.1425%（可打折）與證交稅（ETF 0.1%／個股 0.3%）
- 分「樣本內／樣本外」（預設以 2019-01-01 切分），檢查是否過度擬合

## 使用

```bash
pip install pandas numpy yfinance tabulate
python backtest.py                         # 0050
python backtest.py --ticker 006208.TW
python backtest.py --ticker 2330.TW --stock --fee-discount 0.28
python backtest.py --csv data.csv          # 自備資料：Date、Close（還原權息價）
python backtest.py --demo                  # 模擬資料，僅驗證程式
```

## 怎麼看結果

- **每筆期望值**：扣成本後平均每筆報酬，> 0 才有優勢
- **獲利因子**：總獲利 ÷ 總虧損，> 1.5 較有意義
- **報酬/回撤**：年化報酬 ÷ 最大回撤，越高代表承受的痛苦越值得
- 樣本內與樣本外表現落差大 → 可能是運氣或過度擬合

僅供研究，非投資建議。
