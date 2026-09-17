"""
Optional: pull real 10-K filings from SEC EDGAR so you can run the system over
actual public documents instead of the bundled synthetic samples.

    python ingest/fetch_filings.py AAPL MSFT

EDGAR is free but asks that you identify yourself in the User-Agent header, so
set EDGAR_UA to something like "Your Name your@email.com" before running. This
hits the network, so it won't work in a locked-down sandbox - run it locally.

After fetching, point build_index at data/sample_filings (or wherever you save)
and rebuild the index.
"""
import os
import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import requests

from config import DOCS_DIR

UA = os.getenv("EDGAR_UA", "governed-financial-rag example@example.com")
HEADERS = {"User-Agent": UA}


def cik_for_ticker(ticker: str) -> str:
    # EDGAR keeps a ticker -> CIK map; CIK must be zero-padded to 10 digits
    r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=HEADERS, timeout=30)
    r.raise_for_status()
    for row in r.json().values():
        if row["ticker"].lower() == ticker.lower():
            return str(row["cik_str"]).zfill(10)
    raise ValueError(f"ticker not found on EDGAR: {ticker}")


def latest_10k_text(cik: str) -> str:
    subs = requests.get(
        f"https://data.sec.gov/submissions/CIK{cik}.json", headers=HEADERS, timeout=30
    ).json()
    recent = subs["filings"]["recent"]
    for form, acc, doc in zip(recent["form"], recent["accessionNumber"], recent["primaryDocument"]):
        if form == "10-K":
            acc_nodash = acc.replace("-", "")
            url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}/{doc}"
            html = requests.get(url, headers=HEADERS, timeout=60).text
            # crude tag strip - fine for a demo; use a real HTML parser for prod
            import re
            text = re.sub(r"<[^>]+>", " ", html)
            return re.sub(r"\s+", " ", text)
    raise ValueError(f"no 10-K found for CIK {cik}")


def main(tickers):
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for t in tickers:
        print(f"fetching {t} ...")
        cik = cik_for_ticker(t)
        text = latest_10k_text(cik)
        out = DOCS_DIR / f"{t.lower()}_10k.txt"
        out.write_text(text, encoding="utf-8")
        print(f"  saved {len(text):,} chars -> {out}")
        time.sleep(0.5)  


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python ingest/fetch_filings.py TICKER [TICKER ...]")
    main(sys.argv[1:])
