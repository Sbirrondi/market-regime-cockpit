"""Calcolo delle feature e dei 5 pilastri del regime.

Convenzione fondamentale: ogni feature e' orientata in modo che
**valore alto = RISK-ON / condizioni favorevoli**.
Le feature che nascono al contrario (VIX, spread di credito, drawdown)
vengono invertite di segno all'origine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import allinea_macro

TRADING_YEAR = 252
Z_WINDOW = 3 * TRADING_YEAR      # finestra di normalizzazione: 3 anni
Z_MIN_OBS = TRADING_YEAR         # servono almeno 12 mesi per uno z-score


# --------------------------------------------------------------- utility
def _z(s: pd.Series, window: int = Z_WINDOW, clip: float = 3.0) -> pd.Series:
    """Z-score rolling, senza look-ahead: usa solo il passato fino a t incluso."""
    mu = s.rolling(window, min_periods=Z_MIN_OBS).mean()
    sd = s.rolling(window, min_periods=Z_MIN_OBS).std()
    return ((s - mu) / sd.replace(0, np.nan)).clip(-clip, clip)


def _pct_rank(s: pd.Series, window: int = Z_WINDOW) -> pd.Series:
    """Percentile rolling (0-100) del valore corrente nella sua storia recente."""
    return s.rolling(window, min_periods=Z_MIN_OBS).rank(pct=True) * 100


def _mom(s: pd.Series, n: int) -> pd.Series:
    return s / s.shift(n) - 1.0


def _ratio(px: pd.DataFrame, a: str, b: str) -> pd.Series | None:
    if a not in px.columns or b not in px.columns:
        return None
    r = (px[a].ffill(limit=5) / px[b].ffill(limit=5)).replace([np.inf, -np.inf], np.nan)
    return r if r.notna().sum() > Z_MIN_OBS else None


def _safe(d: dict, key: str, s: pd.Series | None) -> None:
    if s is not None and s.notna().sum() > Z_MIN_OBS:
        d[key] = s


# ---------------------------------------------------------------- PILASTRI
def pillar_trend(px: pd.DataFrame, sectors: list[str], global_idx: list[str]) -> dict:
    """Pilastro 1 - Tendenza e partecipazione. Alto = trend sano e diffuso."""
    f = {}
    spx = px.get("^GSPC")
    if spx is not None:
        spx = spx.ffill(limit=5)
        _safe(f, "spx_vs_50d", spx / spx.rolling(50).mean() - 1)
        _safe(f, "spx_vs_200d", spx / spx.rolling(200).mean() - 1)
        ma200 = spx.rolling(200).mean()
        _safe(f, "pendenza_200d", ma200 / ma200.shift(63) - 1)
        _safe(f, "drawdown_da_max_1a", spx / spx.rolling(TRADING_YEAR).max() - 1)
        _safe(f, "momentum_12_1", spx.shift(21) / spx.shift(TRADING_YEAR) - 1)

    # ampiezza: quota di settori sopra la propria media a 200 giorni
    have = [c for c in sectors if c in px.columns]
    if len(have) >= 5:
        sub = px[have].ffill(limit=5)
        above = (sub > sub.rolling(200).mean()).astype(float)
        _safe(f, "settori_sopra_200d", above.mean(axis=1) * 100)
        above50 = (sub > sub.rolling(50).mean()).astype(float)
        _safe(f, "settori_sopra_50d", above50.mean(axis=1) * 100)

    haveg = [c for c in global_idx if c in px.columns]
    if len(haveg) >= 4:
        subg = px[haveg].ffill(limit=5)
        _safe(f, "indici_globali_sopra_200d",
              (subg > subg.rolling(200).mean()).astype(float).mean(axis=1) * 100)

    _safe(f, "ampiezza_equal_weight", _ratio(px, "RSP", "SPY"))
    return f


def pillar_volatility(px: pd.DataFrame) -> dict:
    """Pilastro 2 - Volatilita' e stress. Gia' invertito: alto = calma."""
    f = {}
    vix = px.get("^VIX")
    if vix is not None:
        vix = vix.ffill(limit=5)
        _safe(f, "vix_livello_inv", -vix)
        v3m = px.get("^VIX3M")
        if v3m is not None:                      # contango = mercato tranquillo
            _safe(f, "vix_struttura_termine", (v3m.ffill(limit=5) / vix) - 1)
        v9d = px.get("^VIX9D")
        if v9d is not None:                      # stress a brevissimo
            _safe(f, "vix_stress_immediato", (vix / v9d.ffill(limit=5)) - 1)
        vvix = px.get("^VVIX")
        if vvix is not None:
            _safe(f, "vvix_inv", -vvix.ffill(limit=5))

    move = px.get("^MOVE")
    if move is not None:
        _safe(f, "move_bond_inv", -move.ffill(limit=5))

    spx = px.get("^GSPC")
    if spx is not None:
        ret = spx.ffill(limit=5).pct_change()
        rv = ret.rolling(21).std() * np.sqrt(TRADING_YEAR) * 100
        _safe(f, "vol_realizzata_21g_inv", -rv)
        # NB: il premio implicita-realizzata (VIX meno vol realizzata) e' stato
        # RIMOSSO di proposito. Esplode proprio durante i crolli, perche'
        # l'implicita corre prima della realizzata, e con segno positivo
        # annullava il segnale del VIX: il pilastro risultava quasi cieco al VIX.
        # asimmetria dei rendimenti: coda sinistra pesante = stress
        _safe(f, "asimmetria_63g", ret.rolling(63).skew())

    _safe(f, "vol_oro_inv", -px["^GVZ"].ffill(limit=5) if "^GVZ" in px.columns else None)

    livello = {k: f.pop(k) for k in ("vix_livello_inv", "vol_realizzata_21g_inv")
               if k in f}
    struttura = {k: f.pop(k) for k in ("vix_struttura_termine", "vix_stress_immediato",
                                       "vvix_inv", "asimmetria_63g") if k in f}
    altri = {k: f.pop(k) for k in ("move_bond_inv", "vol_oro_inv") if k in f}
    out = {}
    if livello:
        out["livello"] = livello
    if struttura:
        out["struttura"] = struttura
    if altri:
        out["altre_classi"] = altri
    return out


def pillar_credit(px: pd.DataFrame, macro: pd.DataFrame | None) -> dict:
    """Pilastro 3 - Credito e liquidita'. Alto = credito che collabora."""
    f = {}
    for name, a, b in [("hy_su_ig", "HYG", "LQD"),
                       ("hy_su_treasury", "HYG", "IEF"),
                       ("ig_su_treasury", "LQD", "IEF"),
                       ("em_su_treasury", "EMB", "IEF"),
                       ("loans_su_treasury", "BKLN", "IEF")]:
        r = _ratio(px, a, b)
        if r is not None:
            _safe(f, name, r)
            _safe(f, f"{name}_slancio_63g", _mom(r, 63))

    if macro is not None and not macro.empty:
        m = allinea_macro(macro, px.index)
        if "BAMLH0A0HYM2" in m:
            hy = m["BAMLH0A0HYM2"]
            _safe(f, "hy_oas_inv", -hy)
            _safe(f, "hy_oas_var_20g_inv", -(hy - hy.shift(20)))
        if "BAMLC0A0CM" in m:
            _safe(f, "ig_oas_inv", -m["BAMLC0A0CM"])
        if "BAMLH0A3HYC" in m and "BAMLH0A0HYM2" in m:
            _safe(f, "qualita_credito_inv", -(m["BAMLH0A3HYC"] - m["BAMLH0A0HYM2"]))
        if "NFCI" in m:
            _safe(f, "condizioni_finanziarie_inv", -m["NFCI"])
        if "STLFSI4" in m:
            _safe(f, "stress_finanziario_inv", -m["STLFSI4"])
        # liquidita' netta = bilancio Fed - conto del Tesoro - reverse repo
        if {"WALCL", "WTREGEN", "RRPONTSYD"} <= set(m.columns):
            netliq = m["WALCL"] - m["WTREGEN"] - m["RRPONTSYD"] * 1000
            _safe(f, "liquidita_netta_var_13sett", netliq - netliq.shift(65))

    spread = {k: f.pop(k) for k in ("hy_oas_inv", "hy_oas_var_20g_inv", "ig_oas_inv",
                                    "qualita_credito_inv") if k in f}
    condizioni = {k: f.pop(k) for k in ("condizioni_finanziarie_inv",
                                        "stress_finanziario_inv",
                                        "liquidita_netta_var_13sett") if k in f}
    out = {}
    if spread:
        out["spread"] = spread
    if condizioni:
        out["condizioni"] = condizioni
    if f:
        out["rapporti_etf"] = dict(f)
    return out


def pillar_macro(px: pd.DataFrame, macro: pd.DataFrame | None) -> dict:
    """Pilastro 4 - Ciclo macro implicito nei prezzi (+ dati veri se disponibili).

    Restituisce due blocchi separati: spinta alla CRESCITA e spinta all'INFLAZIONE.
    """
    growth, infl = {}, {}

    _safe(growth, "rame_su_oro", _ratio(px, "HG=F", "GC=F"))
    _safe(growth, "ciclici_su_difensivi", _ratio(px, "XLY", "XLP"))
    _safe(growth, "small_su_large", _ratio(px, "IWM", "SPY"))
    _safe(growth, "emergenti_su_usa", _ratio(px, "EEM", "SPY"))
    _safe(growth, "semis_su_mercato", _ratio(px, "SMH", "SPY"))
    _safe(growth, "banche_su_mercato", _ratio(px, "XLF", "SPY"))
    _safe(growth, "utilities_su_mercato_inv",
          -_ratio(px, "XLU", "SPY") if _ratio(px, "XLU", "SPY") is not None else None)

    # curva dei tassi: irripidimento con tassi che salgono = crescita attesa
    if {"^TNX", "^IRX"} <= set(px.columns):
        _safe(growth, "curva_10a_3m", px["^TNX"].ffill(limit=5) - px["^IRX"].ffill(limit=5))
    if {"^TNX", "^FVX"} <= set(px.columns):
        _safe(growth, "curva_10a_5a", px["^TNX"].ffill(limit=5) - px["^FVX"].ffill(limit=5))

    _safe(infl, "breakeven_proxy", _ratio(px, "TIP", "IEF"))
    if "CL=F" in px.columns:
        _safe(infl, "slancio_petrolio_126g", _mom(px["CL=F"].ffill(limit=5), 126))
    if "DBC" in px.columns:
        _safe(infl, "slancio_commodity_126g", _mom(px["DBC"].ffill(limit=5), 126))
    if "HG=F" in px.columns:
        _safe(infl, "slancio_rame_126g", _mom(px["HG=F"].ffill(limit=5), 126))
    if "DX-Y.NYB" in px.columns:                 # dollaro debole = inflazione importata
        _safe(infl, "dollaro_inv", -px["DX-Y.NYB"].ffill(limit=5))
    if "^TNX" in px.columns:
        _safe(infl, "tassi_lunghi", px["^TNX"].ffill(limit=5))

    if macro is not None and not macro.empty:
        m = allinea_macro(macro, px.index)
        if "USSLIND" in m:
            _safe(growth, "leading_index_usa", m["USSLIND"])
        if "ICSA" in m:
            _safe(growth, "sussidi_inv", -m["ICSA"].rolling(4).mean())
        if "INDPRO" in m:
            _safe(growth, "produzione_ind_var_annua", m["INDPRO"] / m["INDPRO"].shift(252) - 1)
        if "PAYEMS" in m:
            _safe(growth, "occupazione_slancio_3m", m["PAYEMS"] - m["PAYEMS"].shift(63))
        if "UNRATE" in m:                        # regola di Sahm semplificata
            u = m["UNRATE"]
            _safe(growth, "sahm_inv", -(u.rolling(63).mean() - u.rolling(252).min()))
        if "RSAFS" in m:
            _safe(growth, "vendite_dettaglio_var_annua", m["RSAFS"] / m["RSAFS"].shift(252) - 1)
        if "UMCSENT" in m:
            _safe(growth, "fiducia_consumatori", m["UMCSENT"])
        if "PERMIT" in m:
            _safe(growth, "permessi_edilizi_var_annua", m["PERMIT"] / m["PERMIT"].shift(252) - 1)

        if "T10YIE" in m:
            _safe(infl, "breakeven_10a", m["T10YIE"])
        if "T5YIFR" in m:
            _safe(infl, "inflazione_attesa_5a5a", m["T5YIFR"])
        if "CPILFESL" in m:
            _safe(infl, "cpi_core_var_annua", m["CPILFESL"] / m["CPILFESL"].shift(252) - 1)
        if "PCEPILFE" in m:
            _safe(infl, "pce_core_var_annua", m["PCEPILFE"] / m["PCEPILFE"].shift(252) - 1)
    return {"crescita": growth, "inflazione": infl}


def pillar_risk_appetite(px: pd.DataFrame) -> dict:
    """Pilastro 5 - Propensione al rischio trasversale."""
    f = {}
    for name, a, b in [("azioni_su_bond", "SPY", "TLT"),
                       ("biotech_su_salute", "XBI", "XLV"),
                       ("growth_su_value", "IWF", "IWD"),
                       ("minvol_su_mercato_inv", "USMV", "SPY"),
                       ("oro_su_azioni_inv", "GC=F", "SPY"),
                       ("nasdaq_su_spx", "QQQ", "SPY")]:
        r = _ratio(px, a, b)
        if r is None:
            continue
        if name.endswith("_inv"):
            r = -r
        _safe(f, name, r)

    if "AUDJPY=X" in px.columns:
        _safe(f, "audjpy_carry", px["AUDJPY=X"].ffill(limit=5))
    if "USDCHF=X" in px.columns:
        _safe(f, "franco_rifugio_inv", px["USDCHF=X"].ffill(limit=5))
    if "BTC-USD" in px.columns:
        _safe(f, "bitcoin_slancio_63g", _mom(px["BTC-USD"].ffill(limit=5), 63))
    if {"BTC-USD", "SPY"} <= set(px.columns):
        _safe(f, "bitcoin_su_azioni", _ratio(px, "BTC-USD", "SPY"))
    return f


# ------------------------------------------------------- assemblaggio finale
def _score_block(features: dict, smooth: int = 5) -> tuple[pd.Series, pd.DataFrame]:
    """Punteggio del pilastro in [-100, +100].

    Se `features` contiene sotto-dizionari, ogni sotto-blocco tematico pesa
    uguale. Serve a evitare la diluizione: senza, un pilastro con 13 rapporti
    fra ETF e 4 spread di credito darebbe agli spread un peso di 4/17, quando
    sono la variabile piu' informativa del blocco.
    """
    if not features:
        return pd.Series(dtype=float), pd.DataFrame()
    blocchi, piatte = {}, {}
    for k, v in features.items():
        (blocchi if isinstance(v, dict) else piatte)[k] = v
    zs = pd.DataFrame({k: _z(v.astype(float)) for k, v in piatte.items()})
    medie = [zs.mean(axis=1, skipna=True)] if len(zs.columns) else []
    for nome, sotto in blocchi.items():
        zb = pd.DataFrame({k: _z(v.astype(float)) for k, v in sotto.items()})
        if not len(zb.columns):
            continue
        zs = pd.concat([zs, zb], axis=1)
        medie.append(zb.mean(axis=1, skipna=True))
    if not medie:
        return pd.Series(dtype=float), pd.DataFrame()
    raw = pd.concat(medie, axis=1).mean(axis=1, skipna=True)
    if smooth > 1:
        raw = raw.rolling(smooth, min_periods=1).mean()
    # tanh comprime le code senza troncarle bruscamente
    score = np.tanh(raw / 1.5) * 100
    return score, zs


def build_features(px: pd.DataFrame, macro: pd.DataFrame | None,
                   sectors: list[str], global_idx: list[str]) -> dict:
    px = px.sort_index()
    blocks = {
        "tendenza": pillar_trend(px, sectors, global_idx),
        "volatilita": pillar_volatility(px),
        "credito": pillar_credit(px, macro),
        "propensione_rischio": pillar_risk_appetite(px),
    }
    macro_blocks = pillar_macro(px, macro)

    scores, zdetail = {}, {}
    for name, feats in blocks.items():
        s, z = _score_block(feats)
        if not s.empty:
            scores[name] = s
            zdetail[name] = z
    for name, feats in macro_blocks.items():
        s, z = _score_block(feats, smooth=10)     # il macro va letto piu' lento
        if not s.empty:
            scores[name] = s
            zdetail[name] = z

    return {"scores": pd.DataFrame(scores), "z": zdetail,
            "features": {**blocks, **macro_blocks}}
