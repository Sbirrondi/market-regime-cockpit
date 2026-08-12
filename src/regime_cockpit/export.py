"""Costruzione del payload JSON che alimenta la dashboard."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from . import backtest as B
from . import regime as R
from . import strategy as ST
from . import triggers as TR
from . import validation as VA
from .universe import UNIVERSE, US_SECTORS, US_INDICES, GLOBAL_EQUITY, GLOBAL_ETF, \
    FX, COMMODITIES, CRYPTO, CREDIT_BONDS, US_RATES, VOLATILITY, RATIOS
from .data import FRED_SERIES, SERIE_DISMESSE, allinea_macro

FINESTRE = {"1g": 1, "1sett": 5, "1mese": 21, "3mesi": 63, "6mesi": 126, "1anno": 252}


def _var(px: pd.DataFrame, t: str) -> dict:
    s = px[t].dropna()
    if len(s) < 2:
        return {}
    ultimo = float(s.iloc[-1])
    out = {"ultimo": round(ultimo, 4 if abs(ultimo) < 10 else 2)}
    for nome, n in FINESTRE.items():
        if len(s) > n:
            out[nome] = round((ultimo / float(s.iloc[-1 - n]) - 1) * 100, 2)
    # da inizio anno
    anno = s[s.index >= f"{s.index[-1].year}-01-01"]
    if len(anno) > 1:
        out["ytd"] = round((ultimo / float(anno.iloc[0]) - 1) * 100, 2)
    # posizione nel range a 52 settimane
    a52 = s.tail(252)
    if len(a52) > 60:
        lo, hi = float(a52.min()), float(a52.max())
        out["min52"], out["max52"] = round(lo, 2), round(hi, 2)
        out["pos52"] = round((ultimo - lo) / (hi - lo) * 100, 0) if hi > lo else 50
    # rispetto alla media a 200 giorni
    if len(s) > 200:
        out["vs200"] = round((ultimo / float(s.rolling(200).mean().iloc[-1]) - 1) * 100, 1)
    return out


def blocco(px: pd.DataFrame, mappa: dict) -> list:
    righe = []
    for t in mappa:
        if t not in px.columns:
            continue
        d = _var(px, t)
        if not d:
            continue
        d.update({"ticker": t, "nome": mappa[t][0], "gruppo": mappa[t][2]})
        righe.append(d)
    return righe


def contributi_pilastro(zdetail: dict, pilastro: str, n: int = 6) -> list:
    """Le feature che stanno spingendo di piu' il pilastro, in positivo e negativo."""
    if pilastro not in zdetail:
        return []
    z = zdetail[pilastro]
    if z is None or not len(z.columns):
        return []
    z = z.dropna(how="all")
    if z.empty:
        return []
    ultimo = z.iloc[-1].dropna().sort_values()
    voci = list(ultimo.items())
    scelte = voci[:n // 2] + voci[-(n - n // 2):]
    return [{"nome": k.replace("_", " "), "z": round(float(v), 2)} for k, v in scelte]


def diff_giornaliero(panel: pd.DataFrame, scores: pd.DataFrame,
                     px: pd.DataFrame, macro: pd.DataFrame) -> dict:
    """Cosa è cambiato da ieri. Si legge direttamente dal pannello, così
    funziona anche al primo avvio senza bisogno di uno storico salvato."""
    d = panel.dropna(subset=["punteggio_tattico"])
    if len(d) < 30:
        return {}
    oggi, ieri = d.iloc[-1], d.iloc[-2]
    sett = d.iloc[-6] if len(d) > 6 else ieri

    voci = []
    dp = float(oggi["punteggio_tattico"] - ieri["punteggio_tattico"])
    if abs(dp) >= 0.3:
        voci.append({"tipo": "punteggio", "testo": "Punteggio tattico",
                     "delta": round(dp, 1), "rilievo": min(3, int(abs(dp)))})
    if oggi["stato_tattico"] != ieri["stato_tattico"]:
        voci.append({"tipo": "stato", "testo": f"CAMBIO DI STATO: "
                     f"{ieri['stato_tattico']} → {oggi['stato_tattico']}",
                     "delta": None, "rilievo": 4})
    if oggi.get("direzione") != ieri.get("direzione"):
        voci.append({"tipo": "direzione", "testo": f"La direzione passa a "
                     f"{oggi.get('direzione')}", "delta": None, "rilievo": 2})
    if "quadrante" in d.columns and oggi["quadrante"] != ieri["quadrante"]:
        voci.append({"tipo": "quadrante", "testo": f"QUADRANTE MACRO: "
                     f"{ieri['quadrante']} → {oggi['quadrante']}", "delta": None,
                     "rilievo": 4})

    nomi = {"tendenza": "Tendenza", "volatilita": "Volatilità",
            "credito": "Credito e liquidità", "propensione_rischio": "Propensione al rischio",
            "crescita": "Crescita (macro)", "inflazione": "Inflazione (macro)"}
    pil = []
    for c in scores.columns:
        if c not in d.columns or pd.isna(oggi.get(c)) or pd.isna(ieri.get(c)):
            continue
        pil.append({"nome": nomi.get(c, c), "oggi": round(float(oggi[c]), 1),
                    "delta_1g": round(float(oggi[c] - ieri[c]), 1),
                    "delta_5g": round(float(oggi[c] - sett[c]), 1)
                    if not pd.isna(sett.get(c)) else None})
    pil.sort(key=lambda r: -abs(r["delta_1g"]))

    # strumenti che oggi toccano un estremo del proprio range a 5 anni
    estremi = []
    for c in px.columns:
        s = px[c].dropna()
        if len(s) < 1260:
            continue
        v = float(s.iloc[-1])
        coda = s.tail(1260)
        p = float((coda < v).mean() * 100)
        if p >= 99 or p <= 1:
            prec = float((s.tail(1261).iloc[:-1] < float(s.iloc[-2])).mean() * 100)
            estremi.append({"ticker": c, "nome": UNIVERSE.get(c, (c,))[0],
                            "percentile": round(p, 0), "valore": round(v, 2),
                            "nuovo": bool((p >= 99 > prec) or (p <= 1 < prec))})
    estremi.sort(key=lambda r: (not r["nuovo"], -abs(r["percentile"] - 50)))

    return {"data": str(d.index[-1].date()), "data_precedente": str(d.index[-2].date()),
            "eventi": sorted(voci, key=lambda r: -r["rilievo"]),
            "pilastri": pil, "estremi": estremi[:14]}


def costruisci_payload(px: pd.DataFrame, macro: pd.DataFrame,
                       panel: pd.DataFrame, scores: pd.DataFrame,
                       zdetail: dict, settori: list | None = None,
                       globali: list | None = None, completo: bool = True) -> dict:
    snap = R.istantanea(panel, scores)
    valido = panel.dropna(subset=["punteggio_tattico"])

    # ---- storia del punteggio e dell'indice, campionata a settimana
    storia = valido[["punteggio_tattico", "stato_tattico"]].copy()
    storia["spx"] = px["^GSPC"].reindex(storia.index).ffill()
    storia = storia.resample("W-FRI").last().dropna()
    serie_storia = [{"d": str(i.date()), "s": round(float(r.punteggio_tattico), 1),
                     "st": r.stato_tattico, "spx": round(float(r.spx), 1)}
                    for i, r in storia.iterrows()]

    # ---- traiettoria del quadrante macro (ultimi 2 anni, mensile)
    traj = []
    if {"crescita_liscia", "inflazione_liscia"} <= set(panel.columns):
        qq = panel[["crescita_liscia", "inflazione_liscia"]].dropna()
        q = qq[qq.index >= qq.index.max() - pd.DateOffset(months=24)].resample("ME").last()
        traj = [{"d": str(i.date()), "g": round(float(r.crescita_liscia), 1),
                 "i": round(float(r.inflazione_liscia), 1)} for i, r in q.iterrows()]

    # ---- base rate del regime attuale
    b2 = B.base_rate_2d(panel, px["^GSPC"])
    b1 = B.base_rate(panel, px["^GSPC"])
    liv, dirz = snap["tattico"]["stato"], snap["tattico"].get("direzione")
    corrente = b2[(b2["livello"] == liv) & (b2["direzione"] == dirz)]
    base_corrente = corrente.iloc[0].to_dict() if len(corrente) else {}

    # ---- macro: livelli chiave con percentile storico
    macro_rows = []
    if macro is not None and not macro.empty:
        m = allinea_macro(macro, px.index)
        for sid in ["BAMLH0A0HYM2", "BAMLC0A0CM", "BAMLH0A3HYC", "NFCI", "STLFSI4",
                    "DGS10", "DGS2", "DGS3MO", "T10Y2Y", "T10Y3M", "DFII10",
                    "T5YIE", "T10YIE", "T5YIFR", "DFEDTARU", "SOFR",
                    "UNRATE", "ICSA", "PAYEMS", "INDPRO", "RSAFS", "CPILFESL",
                    "PCEPILFE", "UMCSENT", "HOUST", "PERMIT", "M2SL"]:
            if sid not in m.columns:
                continue
            s = m[sid].dropna()
            if len(s) < 60:
                continue
            v = float(s.iloc[-1])
            r = {"id": sid, "nome": FRED_SERIES[sid][0], "gruppo": FRED_SERIES[sid][1],
                 "valore": round(v, 3),
                 "pct_5a": round(float((s.tail(1260) < v).mean() * 100), 0),
                 "pct_storico": round(float((s < v).mean() * 100), 0)}
            for nome, n in [("var_1m", 21), ("var_3m", 63), ("var_12m", 252)]:
                if len(s) > n:
                    r[nome] = round(v - float(s.iloc[-1 - n]), 3)
            # variazione percentuale annua: ha senso solo per serie di livello
            # strettamente positive (un indice che oscilla attorno a zero come
            # NFCI produrrebbe percentuali prive di significato)
            base = float(s.iloc[-1 - 252]) if len(s) > 252 else None
            if (FRED_SERIES[sid][2] in ("level", "index") and base is not None
                    and base > 0 and v > 0 and s.tail(260).min() > 0):
                r["yoy_%"] = round((v / base - 1) * 100, 2)
            macro_rows.append(r)

    # ---- rapporti cross-asset chiave
    ratio_rows = []
    for a, b, nome, tema, _ in RATIOS:
        if a not in px.columns or b not in px.columns:
            continue
        r = (px[a] / px[b]).dropna()
        if len(r) < 300:
            continue
        v = float(r.iloc[-1])
        row = {"nome": nome, "tema": tema, "valore": round(v, 4),
               "pct_5a": round(float((r.tail(1260) < v).mean() * 100), 0)}
        for k, n in [("var_1m", 21), ("var_3m", 63), ("var_6m", 126)]:
            if len(r) > n:
                row[k] = round((v / float(r.iloc[-1 - n]) - 1) * 100, 2)
        ratio_rows.append(row)

    # ---- controllo di freschezza: una serie ferma inquina il pilastro in silenzio
    fine = px.index.max()
    obsolete = []
    for c in px.columns:
        s = px[c].dropna()
        if s.empty:
            continue
        gg = int(np.busday_count(s.index[-1].date(), fine.date()))
        if gg > 5:
            obsolete.append({"ticker": c, "nome": UNIVERSE.get(c, (c,))[0],
                             "ultimo": str(s.index[-1].date()), "giorni_fermo": gg})
    obsolete.sort(key=lambda d: -d["giorni_fermo"])
    if macro is not None and not macro.empty:
        for c in macro.columns:
            s = macro[c].dropna()
            if s.empty or c in SERIE_DISMESSE or len(s) < 200:
                continue
            passo = s.index.to_series().diff().dt.days.median()
            if passo is None or passo > 4:
                continue        # solo le giornaliere: le mensili sono ferme per costruzione
            gg = int(np.busday_count(s.index[-1].date(), fine.date()))
            if gg > 8:
                obsolete.append({"ticker": c, "nome": FRED_SERIES[c][0],
                                 "ultimo": str(s.index[-1].date()), "giorni_fermo": gg})

    lead = B.leadership_settoriale(panel, px, list(US_SECTORS))
    lead_corrente = []
    if not lead.empty and liv in lead.index:
        srt = lead.loc[liv].dropna().sort_values(ascending=False)
        lead_corrente = [{"ticker": k, "nome": US_SECTORS[k][0], "extra_%": float(v)}
                         for k, v in srt.items()]

    return {
        "meta": {
            "generato": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "ultimo_dato": snap["data"],
            "strumenti": int(px.shape[1]),
            "serie_macro": int(macro.shape[1]) if macro is not None else 0,
            "storico_da": str(valido.index.min().date()),
            "serie_obsolete": obsolete,
        },
        "snapshot": snap,
        "pilastri_dettaglio": {p: contributi_pilastro(zdetail, p)
                               for p in scores.columns},
        "storia": serie_storia,
        "traiettoria_macro": traj,
        "base_rate_corrente": base_corrente,
        "base_rate_2d": b2.replace({np.nan: None}).to_dict("records"),
        "base_rate_1d": b1.replace({np.nan: None}).to_dict("records"),
        "transizioni": B.matrice_transizione(panel).to_dict(),
        "stabilita": B.stabilita(panel),
        "leadership": [],   # rimossa: t-stat sotto 1, era rumore presentato come segnale
        "mercati": {
            "indici_usa": blocco(px, US_INDICES),
            "settori": blocco(px, US_SECTORS),
            "globale": blocco(px, {**GLOBAL_EQUITY, **GLOBAL_ETF}),
            "tassi": blocco(px, US_RATES),
            "credito": blocco(px, CREDIT_BONDS),
            "volatilita": blocco(px, VOLATILITY),
            "valute": blocco(px, FX),
            "materie_prime": blocco(px, COMMODITIES),
            "cripto": blocco(px, CRYPTO),
        },
        "macro": macro_rows,
        "rapporti": ratio_rows,
        "episodi": B.episodi_storici(panel, px["^GSPC"], durata_min=20)
                    .replace({np.nan: None}).to_dict("records"),
        "diff": diff_giornaliero(panel, scores, px, macro),
        "trigger": TR.calcola(px, macro, panel, settori or list(US_SECTORS),
                              globali or list(GLOBAL_EQUITY)) if completo else {},
        "strategia": ST.esegui(px, panel) if completo else {},
        "validazione": VA.esegui(panel, px) if completo else {},
    }


def salva(payload: dict, path: str) -> None:
    with open(path, "w", encoding="utf8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"), default=str)
