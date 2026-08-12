"""Fetch dati di mercato.

Fonte primaria: Yahoo Finance chart API (nessuna chiave richiesta).
Fonte macro: FRED API (richiede chiave gratuita in FRED_API_KEY).

Tutto viene messo in cache su disco in formato parquet/csv, cosi' ad ogni
run si costruisce lo storico invece di ripartire da zero.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

CACHE = Path(__file__).resolve().parents[2] / "data" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                     "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}

# 1° gennaio 1990: piu' indietro di cosi' pochi strumenti hanno dati utili
START_EPOCH = 631152000


def _get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers=_UA)
    return urllib.request.urlopen(req, timeout=timeout).read()


# --------------------------------------------------------------- YAHOO
def fetch_yahoo(ticker: str, range_: str = "10y", interval: str = "1d",
                retries: int = 6) -> pd.DataFrame | None:
    """Serie OHLCV giornaliera da Yahoo.

    Yahoo limita per IP: su 429 si aspetta con backoff esponenziale e si
    alterna fra gli host query1/query2. Ritorna None se il ticker non risponde.
    """
    # NB: range=max con interval=1d viene aggregato a mensile da Yahoo.
    # Per lo storico giornaliero completo servono period1/period2 espliciti.
    if range_ == "max":
        span = f"period1={START_EPOCH}&period2={int(time.time())}"
    else:
        span = f"range={range_}"

    hosts = ("query2", "query1")
    last_err = None
    for attempt in range(retries):
        host = hosts[attempt % 2]
        url = (f"https://{host}.finance.yahoo.com/v8/finance/chart/"
               f"{urllib.parse.quote(ticker)}?{span}&interval={interval}"
               "&includePrePost=false&events=div%2Csplit")
        try:
            raw = json.loads(_get(url))
            res = raw["chart"]["result"]
            if not res:
                return None
            r = res[0]
            ts = r.get("timestamp")
            if not ts:
                return None
            q = r["indicators"]["quote"][0]
            df = pd.DataFrame({
                "open": q.get("open"),
                "high": q.get("high"),
                "low": q.get("low"),
                "close": q.get("close"),
                "volume": q.get("volume"),
            }, index=pd.to_datetime(ts, unit="s", utc=True).tz_convert("UTC")
                 .normalize().tz_localize(None))
            adj = r["indicators"].get("adjclose")
            df["adjclose"] = adj[0]["adjclose"] if adj else df["close"]
            df.index.name = "date"
            df = df[~df.index.duplicated(keep="last")].sort_index()
            return df.dropna(subset=["close"])
        except Exception as e:          # noqa: BLE001
            last_err = e
            is_429 = "429" in str(e)
            wait = (12 * 2 ** attempt) if is_429 else (2 * (attempt + 1))
            time.sleep(min(wait, 240))
    print(f"  [warn] {ticker}: {type(last_err).__name__} {last_err}", flush=True)
    return None


def build_price_panel(tickers: list[str], range_: str = "10y",
                      cache_name: str = "prices.parquet",
                      use_cache: bool = True) -> pd.DataFrame:
    """Pannello wide di prezzi adjusted-close. Merge incrementale con la cache."""
    path = CACHE / cache_name
    old = None
    if use_cache and path.exists():
        try:
            old = pd.read_parquet(path)
        except Exception:                # noqa: BLE001
            old = None

    # riprende da dove si era fermato: salta i ticker gia' aggiornati oggi
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    frames, meta = {}, {}
    meta_path = CACHE / "fetch_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:                # noqa: BLE001
            meta = {}

    todo = []
    for t in tickers:
        fresh = (old is not None and t in old.columns
                 and old[t].dropna().index.max() >= today - pd.Timedelta(days=1))
        if fresh and not os.environ.get("FORCE_REFETCH"):
            continue
        todo.append(t)
    print(f"  da scaricare: {len(todo)}/{len(tickers)}", flush=True)

    pace = float(os.environ.get("YF_PACE", "3.0"))
    for i, t in enumerate(todo, 1):
        df = fetch_yahoo(t, range_=range_)
        if df is not None and not df.empty:
            frames[t] = df["adjclose"]
            meta[t] = {"last": float(df["adjclose"].iloc[-1]),
                       "last_date": str(df.index[-1].date()),
                       "n_obs": int(len(df))}
            print(f"  [{i:>3}/{len(todo)}] {t:14s} {len(df):>6} oss.  "
                  f"ultimo {df.index[-1].date()} = {df['adjclose'].iloc[-1]:,.2f}", flush=True)
        # salvataggio incrementale ogni 10 ticker: un 429 non fa perdere il lavoro
        if i % 10 == 0 and frames:
            part = pd.DataFrame(frames).sort_index()
            base = old if old is not None else pd.DataFrame()
            merged = part.combine_first(base) if not base.empty else part
            for c in base.columns:
                if c not in merged.columns:
                    merged[c] = base[c]
            merged.index = pd.to_datetime(merged.index).normalize()
            merged[~merged.index.duplicated(keep="last")].to_parquet(path)
            meta_path.write_text(json.dumps(meta, indent=1))
        time.sleep(pace)

    new = pd.DataFrame(frames).sort_index() if frames else pd.DataFrame()
    if old is not None and not old.empty:
        new = new.combine_first(old).sort_index()
        for c in old.columns:
            if c not in new.columns:
                new[c] = old[c]

    new.index = pd.to_datetime(new.index).normalize()
    new = new[~new.index.duplicated(keep="last")]
    new.to_parquet(path)
    (CACHE / "fetch_meta.json").write_text(json.dumps(meta, indent=1))
    return new


# ---------------------------------------------------------------- FRED
FRED_SERIES = {
    # credito / stress
    "BAMLH0A0HYM2": ("HY OAS", "credit", "bps_like"),
    "BAMLC0A0CM": ("IG OAS", "credit", "bps_like"),
    "BAMLH0A3HYC": ("CCC & lower OAS", "credit", "bps_like"),
    "NFCI": ("Chicago Fed NFCI", "conditions", "index"),
    "ANFCI": ("NFCI adjusted", "conditions", "index"),
    "STLFSI4": ("St.Louis Financial Stress", "conditions", "index"),
    # liquidita'
    "WALCL": ("Fed balance sheet", "liquidity", "level"),
    "RRPONTSYD": ("Reverse repo ON", "liquidity", "level"),
    "WTREGEN": ("Treasury General Account", "liquidity", "level"),
    "M2SL": ("M2", "liquidity", "level"),
    # tassi / curva / inflazione attesa
    "DGS10": ("UST 10Y", "rates", "pct"),
    "DGS2": ("UST 2Y", "rates", "pct"),
    "DGS3MO": ("UST 3M", "rates", "pct"),
    "T10Y2Y": ("Curva 10Y-2Y", "rates", "pct"),
    "T10Y3M": ("Curva 10Y-3M", "rates", "pct"),
    "DFII10": ("Tasso reale 10Y", "rates", "pct"),
    "T5YIE": ("Breakeven 5Y", "inflation", "pct"),
    "T10YIE": ("Breakeven 10Y", "inflation", "pct"),
    "T5YIFR": ("5y5y forward inflation", "inflation", "pct"),
    "DFEDTARU": ("Fed funds target (alto)", "policy", "pct"),
    "SOFR": ("SOFR", "policy", "pct"),
    # macro reale
    "UNRATE": ("Tasso disoccupazione", "growth", "pct"),
    "ICSA": ("Richieste sussidi", "growth", "level"),
    "PAYEMS": ("Occupati non-agricoli", "growth", "level"),
    "INDPRO": ("Produzione industriale", "growth", "index"),
    "RSAFS": ("Vendite al dettaglio", "growth", "level"),
    "CPIAUCSL": ("CPI", "inflation", "index"),
    "CPILFESL": ("CPI core", "inflation", "index"),
    "PCEPILFE": ("PCE core", "inflation", "index"),
    "UMCSENT": ("Fiducia consumatori Michigan", "growth", "index"),
    "HOUST": ("Cantieri avviati", "growth", "level"),
    "PERMIT": ("Permessi edilizi", "growth", "level"),
    "USSLIND": ("Leading Index USA", "growth", "pct"),
    "USREC": ("Recessione NBER", "growth", "flag"),
}


def fetch_fred(series_id: str, api_key: str, start: str = "1995-01-01") -> pd.Series | None:
    url = ("https://api.stlouisfed.org/fred/series/observations"
           f"?series_id={series_id}&api_key={api_key}&file_type=json"
           f"&observation_start={start}")
    try:
        raw = json.loads(_get(url, timeout=40))
        obs = raw.get("observations", [])
        if not obs:
            return None
        s = pd.Series(
            {pd.Timestamp(o["date"]): (float(o["value"]) if o["value"] not in (".", "") else None)
             for o in obs}, name=series_id, dtype="float64")
        return s.dropna().sort_index()
    except Exception as e:               # noqa: BLE001
        print(f"  [warn] FRED {series_id}: {type(e).__name__} {e}")
        return None


def build_macro_panel(api_key: str | None = None,
                      cache_name: str = "macro.parquet") -> pd.DataFrame:
    api_key = api_key or os.environ.get("FRED_API_KEY", "")
    path = CACHE / cache_name
    if not api_key:
        print("  [info] nessuna FRED_API_KEY -> pannello macro saltato")
        if path.exists():
            return pd.read_parquet(path)
        return pd.DataFrame()

    out = {}
    for sid in FRED_SERIES:
        s = fetch_fred(sid, api_key)
        if s is not None:
            out[sid] = s
        time.sleep(0.25)
    df = pd.DataFrame(out).sort_index()
    df.to_parquet(path)
    return df


# --------------------------------------------------- RITARDO DI PUBBLICAZIONE
# Il CPI di giugno non esiste il 1 giugno: esce a meta' luglio. Usare la data
# del periodo di riferimento inserirebbe informazione dal futuro nel backtest.
# Qui si ritarda ogni serie di quanto realmente serve perche' sia pubblica.
LAG_PUBBLICAZIONE = {
    # mensili: escono fra 2 e 6 settimane dopo la fine del mese di riferimento
    "CPIAUCSL": 45, "CPILFESL": 45, "PCEPILFE": 58, "INDPRO": 47,
    "RSAFS": 46, "PAYEMS": 38, "UNRATE": 38, "HOUST": 48, "PERMIT": 48,
    "UMCSENT": 40, "USREC": 400,          # la datazione NBER arriva con anni di ritardo
    # settimanali
    "ICSA": 5, "NFCI": 8, "ANFCI": 8, "STLFSI4": 6,
    "WALCL": 8, "WTREGEN": 3, "M2SL": 55,
    # giornaliere di mercato: disponibili il giorno stesso o con 1 giorno
    "BAMLH0A0HYM2": 1, "BAMLC0A0CM": 1, "BAMLH0A3HYC": 1,
    "DGS10": 1, "DGS2": 1, "DGS3MO": 1, "T10Y2Y": 1, "T10Y3M": 1,
    "DFII10": 1, "T5YIE": 1, "T10YIE": 1, "T5YIFR": 1,
    "DFEDTARU": 1, "SOFR": 2, "RRPONTSYD": 2,
}

# serie interrotte: non vanno usate come feature viva
SERIE_DISMESSE = {"USSLIND"}


def allinea_macro(macro: pd.DataFrame, indice: pd.DatetimeIndex) -> pd.DataFrame:
    """Riporta le serie macro sul calendario di mercato rispettando il ritardo
    con cui sono state effettivamente pubblicate."""
    if macro is None or macro.empty:
        return pd.DataFrame(index=indice)
    if macro.index.equals(indice):
        return macro          # gia' allineato: evita di rifare il lavoro
    out = {}
    for col in macro.columns:
        if col in SERIE_DISMESSE:
            continue
        s = macro[col].dropna()
        if s.empty:
            continue
        lag = LAG_PUBBLICAZIONE.get(col, 30)
        s.index = s.index + pd.Timedelta(days=lag)
        out[col] = s[~s.index.duplicated(keep="last")].reindex(
            indice.union(s.index)).ffill().reindex(indice)
    return pd.DataFrame(out, index=indice)
