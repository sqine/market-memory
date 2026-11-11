import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = {"S&P 500": "^GSPC", "NASDAQ": "^IXIC", "Apple": "AAPL", "Bitcoin": "BTC-USD", "KOSPI": "^KS11",
           "Samsung": "005930.KS", "Google": "GOOGL", "Intel": "INTC", "cocacola": "KO", "nvdia": "NVDA",
           "tesla": "TSLA", "sk hynix": "000660.KS", "eth": "ETH-USD"}


def main(end="2024-12-31"):
    out = ROOT / "data" / "raw" / "hurst"
    out.mkdir(parents=True, exist_ok=True)
    for unit in ["1d", "1wk", "1mo"]:
        for name, sym in SYMBOLS.items():
            df = yf.Ticker(sym).history(start="1980-01-01", end=end, interval=unit)
            df["Close"].dropna().to_csv(out / f"{name}_{unit}.csv")
    chaos = ROOT / "data" / "raw" / "chaos"
    chaos.mkdir(parents=True, exist_ok=True)
    g = yf.download("GC=F", start="1990-01-01", progress=False, auto_adjust=False)["Close"].squeeze().dropna()
    g.rename("Price").rename_axis("Date").to_csv(chaos / "GC=F.csv")
    cpi = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL", parse_dates=[0])
    cpi.columns = ["DATE", "CPIAUCSL"]
    cpi[cpi["DATE"] >= "1990-01-01"].to_csv(chaos / "cpi_data.csv", index=False)


if __name__ == "__main__":
    main(*sys.argv[1:])
