"""Dal regime alla decisione: quanto rischio avere addosso.

Premessa dimostrata altrove (validation.py): il regime NON predice i rendimenti,
predice la volatilità. Quindi l'uso corretto non è "compra o vendi" ma
"dimensiona l'esposizione inversamente alla volatilità attesa".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

H = 63                  # orizzonte della previsione di volatilità
VOL_TARGET = 0.12       # 12% annuo: il livello di rischio che si vuole tenere
CAP = 1.5               # leva massima
BANDA = 0.05            # si ribilancia solo oltre 5 punti di scostamento
COSTO = 0.0005          # 5 bp per unità di turnover


def vol_forward(ret: pd.Series, h: int = H) -> pd.Series:
    """Volatilità annualizzata realizzata nei prossimi h giorni. Nota solo dopo."""
    return ret.rolling(h).std().shift(-h) * np.sqrt(252)


def previsione_vol(ret: pd.Series, stati: pd.Series, h: int = H,
                   min_storia: int = 756) -> pd.DataFrame:
    """Tre previsioni della volatilità futura, tutte senza look-ahead.

    - `regime`   : media della vol forward osservata storicamente in questo stato,
                   stimata su finestra espansiva e solo su osservazioni chiuse
    - `trailing` : vol realizzata a 63 giorni, ritardata di un giorno
    - `combinata`: media delle due (è la migliore delle tre, verificato)
    """
    rvf = vol_forward(ret, h)
    idx = stati.index
    reg = pd.Series(index=idx, dtype=float)
    for i in range(len(idx)):
        if i < min_storia:
            continue
        taglio = idx[i] - pd.Timedelta(days=int(h * 1.5))
        storico = rvf[rvf.index < taglio].dropna()
        if len(storico) < 250:
            continue
        campione = storico[stati.reindex(storico.index) == stati.iloc[i]]
        reg.iloc[i] = campione.mean() if len(campione) >= 60 else storico.mean()
    trail = (ret.rolling(63).std() * np.sqrt(252)).shift(1).reindex(idx)
    return pd.DataFrame({"regime": reg, "trailing": trail,
                         "combinata": 0.5 * reg + 0.5 * trail})


def esposizione(prev_vol: pd.Series, target: float = VOL_TARGET,
                cap: float = CAP, banda: float = BANDA) -> pd.Series:
    """Esposizione azionaria con banda di non-intervento sul ribilanciamento."""
    grezza = (target / prev_vol).clip(0, cap)
    fuori, corrente = [], np.nan
    for v in grezza:
        if np.isnan(v):
            fuori.append(np.nan)
            continue
        if np.isnan(corrente) or abs(v - corrente) > banda:
            corrente = v
        fuori.append(corrente)
    return pd.Series(fuori, index=grezza.index)


def statistiche(x: pd.Series) -> dict:
    if len(x) < 60:
        return {}
    n = len(x)
    cagr = (1 + x).prod() ** (252 / n) - 1
    vol = x.std() * np.sqrt(252)
    eq = (1 + x).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    neg = x[x < 0].std() * np.sqrt(252)
    anni = (1 + x).groupby(x.index.year).prod() - 1
    return {
        "cagr_%": round(cagr * 100, 2),
        "vol_%": round(vol * 100, 2),
        "sharpe": round(cagr / vol, 2) if vol else None,
        "sortino": round(cagr / neg, 2) if neg else None,
        "max_dd_%": round(dd * 100, 1),
        "calmar": round(cagr / abs(dd), 2) if dd else None,
        "anno_peggiore_%": round(float(anni.min()) * 100, 1),
        "anni_positivi_%": round(float((anni > 0).mean()) * 100, 0),
    }


def backtest(ret: pd.Series, w: pd.Series, costo: float = COSTO) -> pd.Series:
    w = w.dropna()
    turnover = w.diff().abs().fillna(0)
    return (w * ret.reindex(w.index) - turnover * costo).rename("strategia")


def esegui(px: pd.DataFrame, panel: pd.DataFrame) -> dict:
    """Confronto completo fra buy & hold e le tre varianti di vol-targeting."""
    d = panel.dropna(subset=["punteggio_tattico"])
    spx = px["^GSPC"].reindex(d.index).ffill()
    ret = spx.pct_change().fillna(0)
    prev = previsione_vol(ret, d["stato_tattico"])

    comune = prev["regime"].dropna().index.intersection(prev["trailing"].dropna().index)
    if len(comune) < 500:
        return {}
    bh = ret.reindex(comune)

    varianti, curve = {}, {}
    for col, nome in [("trailing", "Solo vol trailing (nessun regime)"),
                      ("regime", "Solo regime"),
                      ("combinata", "Regime + trailing")]:
        w = esposizione(prev[col].reindex(comune))
        r = backtest(bh, w)
        varianti[col] = {"nome": nome, **statistiche(r),
                         "esposizione_media_%": round(float(w.dropna().mean()) * 100, 0),
                         "turnover_annuo_%": round(float(w.diff().abs().sum()) / len(r) * 252 * 100, 0)}
        curve[col] = (1 + r).cumprod()

    # qualità della previsione di vol: è il test che conta davvero
    rvf = vol_forward(ret).reindex(comune).dropna()
    qualita = {}
    for col in ("regime", "trailing", "combinata"):
        s = prev[col].reindex(rvf.index)
        qualita[col] = {"rmse_pp": round(float(np.sqrt(((s - rvf) ** 2).mean())) * 100, 2),
                        "correlazione": round(float(s.corr(rvf)), 3)}

    # sottoperiodi: un backtest che regge solo in media non regge
    periodi = []
    rc = backtest(bh, esposizione(prev["combinata"].reindex(comune)))
    for lo, hi in [("2000", "2003"), ("2004", "2007"), ("2008", "2009"), ("2010", "2015"),
                   ("2016", "2019"), ("2020", "2022"), ("2023", "2026")]:
        a, b = rc[lo:hi], bh[lo:hi]
        if len(a) < 120:
            continue
        sa, sb = statistiche(a), statistiche(b)
        periodi.append({"periodo": f"{lo}-{hi}", "sharpe": sa.get("sharpe"),
                        "sharpe_bh": sb.get("sharpe"), "max_dd_%": sa.get("max_dd_%"),
                        "max_dd_bh_%": sb.get("max_dd_%")})

    # curva equity campionata a settimana per la dashboard
    eq = pd.DataFrame({"strategia": curve["combinata"],
                       "buy_hold": (1 + bh).cumprod()}).resample("W-FRI").last().dropna()
    serie_eq = [{"d": str(i.date()), "s": round(float(r.strategia), 4),
                 "b": round(float(r.buy_hold), 4)} for i, r in eq.iterrows()]

    w_ora = esposizione(prev["combinata"])
    ultimo = w_ora.dropna()
    stato_ora = d["stato_tattico"].iloc[-1]
    return {
        "oggi": {
            "esposizione_%": round(float(ultimo.iloc[-1]) * 100, 0),
            "vol_attesa_%": round(float(prev["combinata"].dropna().iloc[-1]) * 100, 1),
            "vol_da_regime_%": round(float(prev["regime"].dropna().iloc[-1]) * 100, 1),
            "vol_trailing_%": round(float(prev["trailing"].dropna().iloc[-1]) * 100, 1),
            "stato": stato_ora,
            "target_vol_%": VOL_TARGET * 100,
            "storia": [{"quando": lab,
                        "esposizione_%": round(float(ultimo.iloc[-1 - k]) * 100, 0)}
                       for k, lab in [(5, "1 settimana fa"), (21, "1 mese fa"),
                                      (63, "3 mesi fa"), (252, "1 anno fa")]
                       if len(ultimo) > k],
        },
        "buy_hold": {"nome": "Buy & hold S&P 500", **statistiche(bh),
                     "esposizione_media_%": 100, "turnover_annuo_%": 0},
        "varianti": varianti,
        "qualita_previsione_vol": qualita,
        "sottoperiodi": periodi,
        "equity": serie_eq,
        "campione": {"da": str(comune[0].date()), "a": str(comune[-1].date()),
                     "giorni": len(comune)},
        "parametri": {"vol_obiettivo_%": VOL_TARGET * 100, "leva_max": CAP,
                      "banda_ribilancio_%": BANDA * 100, "costo_bp": COSTO * 10000,
                      "orizzonte_giorni": H},
    }
