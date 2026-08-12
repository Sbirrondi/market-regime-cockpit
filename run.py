"""Punto di ingresso: scarica i dati, ricalcola i regimi, rigenera la dashboard.

Eseguito ogni mattina da GitHub Actions. In locale:
    FRED_API_KEY=... python run.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd

from regime_cockpit.universe import TICKERS, US_SECTORS, GLOBAL_EQUITY
from regime_cockpit.data import build_price_panel, build_macro_panel, CACHE
from regime_cockpit.indicators import build_features
from regime_cockpit import regime as R, export as E

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)


def main() -> None:
    print("1/5 prezzi…", flush=True)
    px = build_price_panel(TICKERS, range_="max")

    print("2/5 macro…", flush=True)
    macro = build_macro_panel()

    px = px.sort_index()
    px = px[px.index >= "1996-01-01"]

    print("3/5 pilastri…", flush=True)
    res = build_features(px, macro, list(US_SECTORS), list(GLOBAL_EQUITY))

    print("4/5 regimi…", flush=True)
    panel = R.costruisci_panel(res["scores"])
    panel.to_parquet(CACHE / "regime_panel.parquet")
    res["scores"].to_parquet(CACHE / "scores.parquet")

    print("5/5 dashboard…", flush=True)
    payload = E.costruisci_payload(px, macro, panel, res["scores"], res["z"],
                                   list(US_SECTORS), list(GLOBAL_EQUITY))
    E.salva(payload, OUT / "payload.json")

    tpl = (Path(__file__).parent / "src" / "regime_cockpit" / "template.html").read_text("utf8")
    html = tpl.replace("__PAYLOAD__", (OUT / "payload.json").read_text("utf8"))
    (OUT / "index.html").write_text(html, encoding="utf8")

    s = payload["snapshot"]
    print(f"\nFATTO — {s['data']}: {s['tattico']['stato']} · {s['tattico']['direzione']} "
          f"({s['tattico']['punteggio']:+.1f})"
          + (f" | macro: {s['allocativo']['quadrante']}" if "allocativo" in s else ""))


if __name__ == "__main__":
    if not os.environ.get("FRED_API_KEY"):
        print("[attenzione] FRED_API_KEY non impostata: il blocco macro sarà assente")
    main()
