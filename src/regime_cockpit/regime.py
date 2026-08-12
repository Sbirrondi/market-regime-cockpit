"""Classificazione del regime: due letture separate.

1. TATTICA (1-4 settimane) - composito di tendenza, volatilita', credito e
   propensione al rischio. Reattiva, dice "come sta respirando il mercato ora".
2. ALLOCATIVA (1-6 mesi) - quadrante crescita/inflazione. Lenta, dice
   "in che parte del ciclo siamo".

La divergenza fra le due e' spesso l'informazione piu' preziosa.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PESI_TATTICI = {
    "tendenza": 0.30,
    "volatilita": 0.25,
    "credito": 0.25,
    "propensione_rischio": 0.20,
}

STATI_TATTICI = [
    (45, "RISK-ON PIENO", "#16a34a",
     "Trend, volatilità e credito remano tutti nella stessa direzione."),
    (15, "COSTRUTTIVO", "#65a30d",
     "Contesto favorevole ma non unanime: si sta lunghi con disciplina."),
    (-15, "NEUTRALE / LATERALE", "#a16207",
     "Nessuna spinta dominante. Il mercato è in attesa di un catalizzatore."),
    (-45, "DIFENSIVO", "#ea580c",
     "Le crepe si allargano: ridurre beta, alzare qualità e liquidità."),
    (-101, "STRESS / RISK-OFF", "#dc2626",
     "Condizioni di stress conclamato. Priorità alla protezione del capitale."),
]

QUADRANTI = {
    ("+", "-"): ("GOLDILOCKS", "#16a34a",
                 "Crescita che tiene, inflazione che scende. Il contesto storicamente "
                 "più generoso per azioni growth e duration insieme."),
    ("+", "+"): ("REFLAZIONE", "#ca8a04",
                 "Crescita e inflazione salgono insieme. Favorisce ciclici, value, "
                 "materie prime, banche; penalizza la duration."),
    ("-", "+"): ("STAGFLAZIONE", "#dc2626",
                 "Crescita che cede con inflazione ancora alta. Il regime piu' ostile: "
                 "storicamente nulla funziona bene tranne beni reali e liquidita'."),
    ("-", "-"): ("CONTRAZIONE", "#2563eb",
                 "Crescita e inflazione in discesa. Favorisce bond lunghi, qualita', "
                 "difensivi; penalizza ciclici e credito rischioso."),
}


def punteggio_tattico(scores: pd.DataFrame) -> pd.Series:
    """Media pesata dei pilastri veloci, ripesata sui pilastri effettivamente presenti."""
    cols = [c for c in PESI_TATTICI if c in scores.columns]
    if not cols:
        return pd.Series(dtype=float)
    w = pd.Series({c: PESI_TATTICI[c] for c in cols})
    sub = scores[cols]
    peso_valido = sub.notna().mul(w, axis=1).sum(axis=1)
    somma = sub.fillna(0).mul(w, axis=1).sum(axis=1)
    out = (somma / peso_valido.replace(0, np.nan))
    return out.rename("punteggio_tattico")


def stato_tattico(score: float) -> tuple[str, str, str]:
    if pd.isna(score):
        return ("DATI INSUFFICIENTI", "#64748b", "Storico non ancora sufficiente.")
    for soglia, nome, colore, desc in STATI_TATTICI:
        if score >= soglia:
            return nome, colore, desc
    return STATI_TATTICI[-1][1:]


# --------------------------------------------------------------- ISTERESI
# Senza isteresi il segnale sfarfalla: con soglie nette un punteggio che
# oscilla attorno a 15 cambia stato ogni due giorni. Si esige che il
# punteggio superi la soglia di un margine prima di riconoscere il cambio.
BUFFER = 6.0

_ORDINE = [s[1] for s in STATI_TATTICI][::-1]        # dal peggiore al migliore
_SOGLIE = [s[0] for s in STATI_TATTICI][::-1]        # soglie inferiori corrispondenti


def stati_con_isteresi(score: pd.Series, buffer: float = BUFFER) -> pd.Series:
    """Sequenza di stati resa stabile da bande di isteresi."""
    s = score.dropna()
    if s.empty:
        return pd.Series(dtype=object)

    def livello(v: float) -> int:
        for i in range(len(_SOGLIE) - 1, -1, -1):
            if v >= _SOGLIE[i]:
                return i
        return 0

    out, corrente = [], livello(s.iloc[0])
    for v in s:
        nuovo = livello(v)
        if nuovo > corrente:
            # per salire serve superare la soglia del livello superiore + buffer
            while corrente < len(_SOGLIE) - 1 and v >= _SOGLIE[corrente + 1] + buffer:
                corrente += 1
        elif nuovo < corrente:
            # per scendere serve bucare la propria soglia di almeno buffer
            while corrente > 0 and v < _SOGLIE[corrente] - buffer:
                corrente -= 1
        out.append(_ORDINE[corrente])
    return pd.Series(out, index=s.index).reindex(score.index)


# --------------------------------------- SECONDA DIMENSIONE: LA DIREZIONE
# Il livello dice DOVE siamo, la direzione dice DOVE STIAMO ANDANDO.
# Storicamente e' la seconda a contenere l'informazione utile: lo stress che
# si riassorbe e lo stress che peggiora hanno esiti opposti a 3 mesi.
def direzione(score: pd.Series, n: int = 21, soglia: float = 8.0) -> pd.Series:
    d = score - score.shift(n)
    return pd.Series(np.where(d > soglia, "IN MIGLIORAMENTO",
                     np.where(d < -soglia, "IN DETERIORAMENTO", "STABILE")),
                     index=score.index)


def quadrante_macro(g: float, i: float) -> tuple[str, str, str]:
    if pd.isna(g) or pd.isna(i):
        return ("DATI INSUFFICIENTI", "#64748b", "Storico non ancora sufficiente.")
    key = ("+" if g >= 0 else "-", "+" if i >= 0 else "-")
    return QUADRANTI[key]


def _persistenza(stati: pd.Series) -> pd.Series:
    """Da quanti giorni consecutivi siamo nello stesso stato."""
    cambio = (stati != stati.shift()).cumsum()
    return stati.groupby(cambio).cumcount() + 1


def costruisci_panel(scores: pd.DataFrame) -> pd.DataFrame:
    """Pannello storico completo del regime, un record per giorno."""
    out = pd.DataFrame(index=scores.index)
    for c in scores.columns:
        out[c] = scores[c]

    # lisciatura in piu' + isteresi: senza, il segnale cambia stato ogni 9 giorni
    grezzo = punteggio_tattico(scores)
    out["punteggio_tattico_grezzo"] = grezzo
    out["punteggio_tattico"] = grezzo.ewm(span=10, min_periods=1).mean()
    out["stato_tattico"] = stati_con_isteresi(out["punteggio_tattico"])
    out["direzione"] = direzione(out["punteggio_tattico"])
    out["stato_completo"] = out["stato_tattico"].fillna("") + " · " + out["direzione"].fillna("")

    if {"crescita", "inflazione"} <= set(scores.columns):
        g = scores["crescita"].rolling(21, min_periods=5).mean()
        i = scores["inflazione"].rolling(21, min_periods=5).mean()
        out["crescita_liscia"] = g
        out["inflazione_liscia"] = i
        out["quadrante"] = [quadrante_macro(a, b)[0] for a, b in zip(g, i)]
        # variazione a 3 mesi: il quadrante sta migliorando o peggiorando?
        out["crescita_delta_63g"] = g - g.shift(63)
        out["inflazione_delta_63g"] = i - i.shift(63)

    # consenso fra i pilastri veloci: bassa dispersione = segnale affidabile
    fast = [c for c in PESI_TATTICI if c in scores.columns]
    if len(fast) >= 3:
        disp = scores[fast].std(axis=1)
        out["consenso"] = (1 - (disp / 70).clip(0, 1)) * 100
        # quanti pilastri concordano col segno del composito
        segno = np.sign(out["punteggio_tattico"])
        concordi = scores[fast].apply(lambda r: (np.sign(r) == segno.loc[r.name]).sum(), axis=1)
        out["pilastri_concordi"] = concordi
        out["pilastri_totali"] = scores[fast].notna().sum(axis=1)

    out["giorni_nello_stato"] = _persistenza(out["stato_tattico"])
    if "quadrante" in out.columns:
        out["giorni_nel_quadrante"] = _persistenza(out["quadrante"])
    return out


def istantanea(panel: pd.DataFrame, scores: pd.DataFrame) -> dict:
    """Fotografia dell'ultimo giorno disponibile, pronta per la dashboard."""
    r = panel.dropna(subset=["punteggio_tattico"]).iloc[-1]
    data = panel.dropna(subset=["punteggio_tattico"]).index[-1]
    nome_t, col_t, desc_t = stato_tattico(r["punteggio_tattico"])

    nome_t = r.get("stato_tattico", nome_t)          # stato con isteresi
    for soglia, nome, colore, desc in STATI_TATTICI:
        if nome == nome_t:
            col_t, desc_t = colore, desc
            break

    snap = {
        "data": str(pd.Timestamp(data).date()),
        "tattico": {
            "punteggio": round(float(r["punteggio_tattico"]), 1),
            "stato": nome_t, "colore": col_t, "descrizione": desc_t,
            "direzione": r.get("direzione"),
            "giorni_nello_stato": int(r.get("giorni_nello_stato", 0)),
            "var_5g": round(float(r["punteggio_tattico"] -
                                  panel["punteggio_tattico"].shift(5).loc[data]), 1)
            if not pd.isna(panel["punteggio_tattico"].shift(5).loc[data]) else None,
            "var_21g": round(float(r["punteggio_tattico"] -
                                   panel["punteggio_tattico"].shift(21).loc[data]), 1)
            if not pd.isna(panel["punteggio_tattico"].shift(21).loc[data]) else None,
        },
        "pilastri": {c: (round(float(r[c]), 1) if not pd.isna(r.get(c, np.nan)) else None)
                     for c in scores.columns},
    }

    if "quadrante" in panel.columns and not pd.isna(r.get("crescita_liscia", np.nan)):
        nome_q, col_q, desc_q = quadrante_macro(r["crescita_liscia"], r["inflazione_liscia"])
        snap["allocativo"] = {
            "quadrante": nome_q, "colore": col_q, "descrizione": desc_q,
            "crescita": round(float(r["crescita_liscia"]), 1),
            "inflazione": round(float(r["inflazione_liscia"]), 1),
            "crescita_delta_63g": round(float(r.get("crescita_delta_63g", np.nan)), 1)
            if not pd.isna(r.get("crescita_delta_63g", np.nan)) else None,
            "inflazione_delta_63g": round(float(r.get("inflazione_delta_63g", np.nan)), 1)
            if not pd.isna(r.get("inflazione_delta_63g", np.nan)) else None,
            "giorni_nel_quadrante": int(r.get("giorni_nel_quadrante", 0)),
        }

    if "consenso" in panel.columns and not pd.isna(r.get("consenso", np.nan)):
        snap["fiducia"] = {
            "consenso": round(float(r["consenso"]), 0),
            "pilastri_concordi": int(r.get("pilastri_concordi", 0)),
            "pilastri_totali": int(r.get("pilastri_totali", 0)),
        }
    return snap
