"""Base rate storiche: cosa e' successo, storicamente, in ciascun regime.

Nessuna previsione. Solo frequenze condizionate misurate sui dati, con
l'avvertenza che il campione per gli stati estremi e' sempre piccolo.

Niente look-ahead: il regime al giorno t usa solo informazione fino a t
(tutti gli z-score sono rolling all'indietro), i rendimenti sono da t a t+n.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ORIZZONTI = {"1 mese": 21, "3 mesi": 63, "6 mesi": 126, "12 mesi": 252}


def _fwd(prezzi: pd.Series, n: int) -> pd.Series:
    return prezzi.shift(-n) / prezzi - 1.0


def _max_drawdown_forward(prezzi: pd.Series, n: int) -> pd.Series:
    """Peggior perdita subita entro i prossimi n giorni, partendo da t."""
    minimo = prezzi[::-1].rolling(n, min_periods=2).min()[::-1]
    return minimo / prezzi - 1.0


def base_rate(panel: pd.DataFrame, prezzi: pd.Series,
              colonna_stato: str = "stato_tattico",
              min_oss: int = 60) -> pd.DataFrame:
    """Rendimenti forward dell'indice condizionati allo stato di regime."""
    px = prezzi.reindex(panel.index).ffill()
    df = pd.DataFrame({"stato": panel[colonna_stato]})
    for nome, n in ORIZZONTI.items():
        df[nome] = _fwd(px, n)
    df["dd_3m"] = _max_drawdown_forward(px, 63)
    df = df.dropna(subset=["stato"])

    righe = []
    for stato, g in df.groupby("stato"):
        r = {"stato": stato, "giorni_osservati": int(len(g)),
             "quota_del_tempo_%": round(100 * len(g) / len(df), 1)}
        for nome in ORIZZONTI:
            s = g[nome].dropna()
            if len(s) < min_oss:
                r[f"{nome}_medio_%"] = None
                r[f"{nome}_positivo_%"] = None
                r[f"{nome}_peggior5%_%"] = None
                continue
            r[f"{nome}_medio_%"] = round(float(s.mean()) * 100, 2)
            r[f"{nome}_mediano_%"] = round(float(s.median()) * 100, 2)
            r[f"{nome}_positivo_%"] = round(float((s > 0).mean()) * 100, 1)
            r[f"{nome}_peggior5%_%"] = round(float(s.quantile(0.05)) * 100, 2)
        dd = g["dd_3m"].dropna()
        r["drawdown_medio_3m_%"] = round(float(dd.mean()) * 100, 2) if len(dd) >= min_oss else None
        r["vol_annua_%"] = round(float(px.pct_change().reindex(g.index).std()
                                       * np.sqrt(252) * 100), 1)
        righe.append(r)
    out = pd.DataFrame(righe)
    return out.sort_values("3 mesi_medio_%", ascending=False, na_position="last")


def leadership_settoriale(panel: pd.DataFrame, px: pd.DataFrame,
                          settori: list[str], benchmark: str = "SPY",
                          orizzonte: int = 63, colonna_stato: str = "stato_tattico",
                          min_oss: int = 60) -> pd.DataFrame:
    """Sovra/sotto-performance media di ciascun settore per stato di regime."""
    have = [s for s in settori if s in px.columns]
    if benchmark not in px.columns or not have:
        return pd.DataFrame()
    bench_f = _fwd(px[benchmark].ffill(), orizzonte)

    out = {}
    for s in have:
        rel = _fwd(px[s].ffill(), orizzonte) - bench_f
        out[s] = rel
    rel_df = pd.DataFrame(out).reindex(panel.index)
    rel_df["stato"] = panel[colonna_stato]

    righe = []
    for stato, g in rel_df.groupby("stato"):
        if len(g) < min_oss:
            continue
        r = {"stato": stato}
        for s in have:
            v = g[s].dropna()
            r[s] = round(float(v.mean()) * 100, 2) if len(v) >= min_oss else None
        righe.append(r)
    return pd.DataFrame(righe).set_index("stato") if righe else pd.DataFrame()


def matrice_transizione(panel: pd.DataFrame, orizzonte: int = 21,
                        colonna_stato: str = "stato_tattico") -> pd.DataFrame:
    """Probabilita' di trovarsi in ciascuno stato fra `orizzonte` giorni."""
    s = panel[colonna_stato].dropna()
    fut = s.shift(-orizzonte)
    tab = pd.crosstab(s, fut, normalize="index") * 100
    return tab.round(1)


def stabilita(panel: pd.DataFrame, colonna_stato: str = "stato_tattico") -> dict:
    """Quanto dura tipicamente un regime e quanto spesso cambia."""
    s = panel[colonna_stato].dropna()
    blocchi = (s != s.shift()).cumsum()
    durate = s.groupby(blocchi).size()
    stati = s.groupby(blocchi).first()
    per_stato = {}
    for stato in s.unique():
        d = durate[stati == stato]
        if len(d):
            per_stato[stato] = {"episodi": int(len(d)),
                                "durata_mediana_giorni": int(d.median()),
                                "durata_massima_giorni": int(d.max())}
    return {"cambi_totali": int(len(durate) - 1),
            "durata_mediana_generale_giorni": int(durate.median()),
            "per_stato": per_stato}


def episodi_storici(panel: pd.DataFrame, prezzi: pd.Series,
                    colonna_stato: str = "stato_tattico",
                    stati_target: tuple = ("STRESS / RISK-OFF", "DIFENSIVO"),
                    durata_min: int = 10) -> pd.DataFrame:
    """Elenca gli episodi di allerta passati: servono a verificare a occhio
    che il modello riconosca 2000, 2008, 2020, 2022."""
    s = panel[colonna_stato].dropna()
    px = prezzi.reindex(s.index).ffill()
    blocchi = (s != s.shift()).cumsum()
    righe = []
    for _, g in s.groupby(blocchi):
        if g.iloc[0] not in stati_target or len(g) < durata_min:
            continue
        ini, fin = g.index[0], g.index[-1]
        seg = px.loc[ini:fin]
        futuro = px.loc[fin:].head(64)
        righe.append({
            "stato": g.iloc[0],
            "inizio": str(ini.date()), "fine": str(fin.date()),
            "giorni": int(len(g)),
            "indice_durante_%": round(float(seg.iloc[-1] / seg.iloc[0] - 1) * 100, 1),
            "min_durante_%": round(float(seg.min() / seg.iloc[0] - 1) * 100, 1),
            "indice_3m_dopo_%": round(float(futuro.iloc[-1] / futuro.iloc[0] - 1) * 100, 1)
            if len(futuro) > 40 else None,
        })
    return pd.DataFrame(righe)


def base_rate_2d(panel: pd.DataFrame, prezzi: pd.Series,
                 min_oss: int = 80) -> pd.DataFrame:
    """La griglia che conta davvero: livello del regime X direzione del regime.

    Serve a rispondere alla domanda giusta. Non "siamo in stress?" ma
    "siamo in stress che peggiora o in stress che si riassorbe?".
    """
    px = prezzi.reindex(panel.index).ffill()
    df = pd.DataFrame({"livello": panel["stato_tattico"],
                       "direzione": panel["direzione"]})
    for nome, n in ORIZZONTI.items():
        df[nome] = _fwd(px, n)
    df["dd_3m"] = _max_drawdown_forward(px, 63)
    df = df.dropna(subset=["livello", "direzione"])
    df = df[df["livello"] != "DATI INSUFFICIENTI"]

    righe = []
    for (liv, dirz), g in df.groupby(["livello", "direzione"]):
        if len(g) < min_oss:
            continue
        r = {"livello": liv, "direzione": dirz, "giorni": int(len(g)),
             "quota_%": round(100 * len(g) / len(df), 1)}
        for nome in ("1 mese", "3 mesi", "6 mesi"):
            s = g[nome].dropna()
            if len(s) < min_oss:
                continue
            r[f"{nome}_medio_%"] = round(float(s.mean()) * 100, 2)
            r[f"{nome}_pos_%"] = round(float((s > 0).mean()) * 100, 1)
        dd = g["dd_3m"].dropna()
        r["dd_3m_%"] = round(float(dd.mean()) * 100, 2) if len(dd) else None
        righe.append(r)
    out = pd.DataFrame(righe)
    if out.empty:
        return out
    return out.sort_values("3 mesi_medio_%", ascending=False, na_position="last")
