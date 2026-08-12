"""Livelli di rottura: a che valore il segnale cambia idea.

Un punteggio senza livelli non è operativo. Qui si fa il problema inverso:
si perturba una variabile osservabile alla volta e si cerca per bisezione il
valore al quale il composito attraverserebbe il confine dello stato.

Si lavora sul punteggio **grezzo** (non sulla media esponenziale) perché la
domanda utile è "a che livello, se ci resta, il regime cambia" — non "quale
shock di un giorno lo fa cambiare", che per costruzione sarebbe enorme.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import allinea_macro
from .indicators import pillar_trend, pillar_volatility, pillar_credit, \
    pillar_risk_appetite, _score_block
from .regime import PESI_TATTICI, STATI_TATTICI, BUFFER

# Alcune variabili non si muovono da sole: alzare lo spread HY tenendo fermo
# quello dei CCC comprimerebbe artificialmente il divario di qualita' e farebbe
# leggere al modello un MIGLIORAMENTO del credito. Qui si muovono insieme.
CO_SHOCK = {"BAMLH0A0HYM2": [("BAMLH0A3HYC", "proporzionale")],
            "^VIX": [("^VIX3M", "proporzionale")]}

# osservabile -> (etichetta, fonte, colonna, pilastri toccati, unità, decimali,
#                 moltiplicatori del range di ricerca)
OSSERVABILI = [
    ("spx", "S&P 500", "px", "^GSPC", ("tendenza", "volatilita"), "punti", 0, (0.45, 1.55)),
    ("vix", "VIX", "px", "^VIX", ("volatilita",), "", 1, (0.30, 6.0)),
    ("hy_oas", "Spread HY (OAS)", "macro", "BAMLH0A0HYM2",
     ("credito",), "%", 2, (0.4, 6.0)),
    ("ig_oas", "Spread IG (OAS)", "macro", "BAMLC0A0CM", ("credito",), "%", 2, (0.4, 6.0)),
    ("nfci", "Condizioni finanziarie NFCI", "macro", "NFCI", ("credito",), "", 3, None),
    ("hyg", "HYG (ETF high yield)", "px", "HYG", ("credito",), "$", 2, (0.70, 1.30)),
    ("rsp", "RSP (S&P equal weight)", "px", "RSP",
     ("tendenza", "propensione_rischio"), "$", 2, (0.65, 1.35)),
    ("tlt", "TLT (Treasury 20+ anni)", "px", "TLT",
     ("credito", "propensione_rischio"), "$", 2, (0.65, 1.35)),
]

_PILASTRO_FN = {
    "tendenza": lambda px, m, s, g: pillar_trend(px, s, g),
    "volatilita": lambda px, m, s, g: pillar_volatility(px),
    "credito": lambda px, m, s, g: pillar_credit(px, m),
    "propensione_rischio": lambda px, m, s, g: pillar_risk_appetite(px),
}


def _composito(valori: dict) -> float:
    """Media pesata dei pilastri, ripesata su quelli disponibili."""
    num = den = 0.0
    for k, w in PESI_TATTICI.items():
        v = valori.get(k)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        num += w * v
        den += w
    return num / den if den else float("nan")


def _confini(stato: str) -> tuple[float | None, float | None]:
    """Soglie da attraversare per scendere / salire di stato, isteresi inclusa."""
    ordine = [s[1] for s in STATI_TATTICI][::-1]      # peggiore -> migliore
    soglie = [s[0] for s in STATI_TATTICI][::-1]
    if stato not in ordine:
        return None, None
    i = ordine.index(stato)
    giu = soglie[i] - BUFFER if i > 0 else None
    su = soglie[i + 1] + BUFFER if i < len(ordine) - 1 else None
    return giu, su


class _Motore:
    """Ricalcola il composito perturbando l'ultimo valore di una serie."""

    def __init__(self, px: pd.DataFrame, macro: pd.DataFrame,
                 settori: list[str], globali: list[str]):
        self.px = px.copy()
        self.macro_all = allinea_macro(macro, px.index) if macro is not None else None
        self.settori, self.globali = settori, globali
        self.base = {k: self._pilastro(k, self.px, self.macro_all)
                     for k in PESI_TATTICI}

    def _pilastro(self, nome: str, px: pd.DataFrame, macro) -> float:
        feats = _PILASTRO_FN[nome](px, macro, self.settori, self.globali)
        s, _ = _score_block(feats)
        return float(s.iloc[-1]) if len(s) and not np.isnan(s.iloc[-1]) else float("nan")

    # Lo shock si applica agli ultimi GIORNI_SHOCK giorni, non a uno solo: i
    # punteggi sono lisciati a 5 giorni, quindi un singolo giorno verrebbe
    # diluito. La domanda operativa e' comunque "se ci arriva e ci RESTA".
    GIORNI_SHOCK = 10

    def con_shock(self, fonte: str, colonna: str, valore: float,
                  pilastri: tuple) -> float:
        px, macro = self.px, self.macro_all
        n = self.GIORNI_SHOCK
        # Lo shock NON appiattisce gli ultimi giorni su un valore costante: li
        # trasla mantenendone la forma, cosi' che uno shock nullo riproduca
        # esattamente il punteggio di partenza e la funzione resti monotona.
        obj = self.px if fonte == "px" else self.macro_all
        v0 = float(obj[colonna].dropna().iloc[-1])
        copia = obj.copy()
        j = copia.columns.get_loc(colonna)
        coda = copia.iloc[-n:, j].astype(float)
        if abs(v0) > 1e-9 and (coda > 0).all():
            k, moltiplicativo = valore / v0, True
            copia.iloc[-n:, j] = coda * k
        else:
            k, moltiplicativo = valore - v0, False
            copia.iloc[-n:, j] = coda + k
        for altra, modo in CO_SHOCK.get(colonna, []):
            if altra not in copia.columns:
                continue
            ja = copia.columns.get_loc(altra)
            coda_a = copia.iloc[-n:, ja].astype(float)
            copia.iloc[-n:, ja] = coda_a * k if moltiplicativo else coda_a + k
        if fonte == "px":
            px = copia
        else:
            macro = copia
        valori = dict(self.base)
        for p in pilastri:
            valori[p] = self._pilastro(p, px, macro)
        return _composito(valori)


def _bisezione(f, lo: float, hi: float, obiettivo: float, iter_max: int = 22):
    """Trova x in [lo,hi] con f(x) ≈ obiettivo. Richiede f monotona sull'intervallo."""
    flo, fhi = f(lo), f(hi)
    if np.isnan(flo) or np.isnan(fhi):
        return None
    if (flo - obiettivo) * (fhi - obiettivo) > 0:
        return None                    # l'obiettivo non è raggiungibile nel range
    for _ in range(iter_max):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if np.isnan(fm):
            return None
        if (flo - obiettivo) * (fm - obiettivo) <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
        if abs(hi - lo) < abs(hi + lo) * 1e-4:
            break
    return 0.5 * (lo + hi)


def calcola(px: pd.DataFrame, macro: pd.DataFrame, panel: pd.DataFrame,
            settori: list[str], globali: list[str]) -> dict:
    stato = panel["stato_tattico"].dropna().iloc[-1]
    grezzo = float(panel["punteggio_tattico_grezzo"].dropna().iloc[-1])
    giu, su = _confini(stato)
    ordine = [s[1] for s in STATI_TATTICI][::-1]
    i = ordine.index(stato) if stato in ordine else None
    stato_giu = ordine[i - 1] if i and i > 0 else None
    stato_su = ordine[i + 1] if i is not None and i < len(ordine) - 1 else None

    motore = _Motore(px, macro, settori, globali)
    righe = []
    for chiave, nome, fonte, col, pilastri, unita, dec, rng in OSSERVABILI:
        serie = (px[col] if fonte == "px" else motore.macro_all[col]).dropna()
        if serie.empty:
            continue
        v0 = float(serie.iloc[-1])
        if rng is None:                      # serie che possono essere negative
            amp = max(abs(v0), float(serie.tail(1260).std()) * 4, 0.5)
            lo, hi = v0 - amp, v0 + amp
        else:
            lo, hi = v0 * rng[0], v0 * rng[1]

        f = lambda x: motore.con_shock(fonte, col, x, pilastri)   # noqa: E731
        r = {"chiave": chiave, "nome": nome, "unita": unita, "decimali": dec,
             "valore_oggi": round(v0, dec + 2),
             "pct_5a": round(float((serie.tail(1260) < v0).mean() * 100), 0)}
        for etichetta, obiettivo, verso in (("giu", giu, stato_giu), ("su", su, stato_su)):
            if obiettivo is None:
                continue
            x = _bisezione(f, lo, hi, obiettivo) or _bisezione(f, hi, lo, obiettivo)
            if x is None:
                continue
            r[etichetta] = {
                "soglia": round(x, dec + 2),
                "verso": verso,
                "distanza_%": round((x / v0 - 1) * 100, 1) if v0 else None,
                "distanza_assoluta": round(x - v0, dec + 2),
            }
        if "giu" in r or "su" in r:
            righe.append(r)

    # ordina per vicinanza: il livello più prossimo è quello che conta oggi
    def prossimita(r):
        d = [abs(r[k]["distanza_%"]) for k in ("giu", "su")
             if k in r and r[k]["distanza_%"] is not None]
        return min(d) if d else 999
    righe.sort(key=prossimita)

    return {
        "stato": stato, "punteggio_grezzo": round(grezzo, 1),
        "soglia_giu": giu, "soglia_su": su,
        "stato_giu": stato_giu, "stato_su": stato_su,
        "pilastri_base": {k: (round(v, 1) if not np.isnan(v) else None)
                          for k, v in motore.base.items()},
        "livelli": righe,
    }
