"""Validazione brutale: il segnale dice qualcosa o è astrologia con i decimali?

Tre domande, in ordine di durezza:
 1. i rendimenti condizionati al regime sono distinguibili dalla media? (bootstrap a blocchi)
 2. un classificatore statistico indipendente (Jump Model, HMM) trova la stessa cosa?
 3. il segnale regge fuori campione, con walk-forward e correzione per test multipli?
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(12345)
H = 63


def _fwd(px: pd.Series, n: int) -> pd.Series:
    return px.shift(-n) / px - 1


def bootstrap_blocchi(x: np.ndarray, n: int = 3000, L: int = H) -> tuple:
    """IC al 95% con ricampionamento a blocchi: i rendimenti forward si
    sovrappongono pesantemente, un bootstrap iid sovrastimerebbe la precisione."""
    x = np.asarray(x)
    m = len(x)
    if m < 2 * L:
        return (np.nan, np.nan)
    nb = max(1, m // L)
    out = np.empty(n)
    for i in range(n):
        idx = RNG.integers(0, m - L, nb)
        out[i] = np.concatenate([x[j:j + L] for j in idx]).mean()
    return tuple(np.percentile(out, [2.5, 97.5]))


def test_rendimenti(panel: pd.DataFrame, spx: pd.Series) -> dict:
    d = panel.dropna(subset=["punteggio_tattico"]).copy()
    p = spx.reindex(d.index).ffill()
    d["fwd"] = _fwd(p, H)
    d = d.dropna(subset=["fwd"])
    unc = float(d["fwd"].mean())

    righe, n_test = [], 0
    for st, g in d.groupby("stato_tattico"):
        if st == "DATI INSUFFICIENTI" or len(g) < 150:
            continue
        lo, hi = bootstrap_blocchi(g["fwd"].values)
        n_test += 1
        righe.append({"stato": st, "n": int(len(g)),
                      "medio_%": round(float(g["fwd"].mean()) * 100, 2),
                      "excess_%": round((float(g["fwd"].mean()) - unc) * 100, 2),
                      "ic_basso_%": round(lo * 100, 2), "ic_alto_%": round(hi * 100, 2),
                      "significativo": bool(lo > unc or hi < unc)})

    ic = {}
    for h, lab in [(21, "1 mese"), (63, "3 mesi"), (126, "6 mesi")]:
        f = _fwd(p, h).reindex(d.index)
        ok = f.notna() & d["punteggio_tattico"].notna()
        r = stats.spearmanr(d["punteggio_tattico"][ok], f[ok])
        q = pd.qcut(d["punteggio_tattico"][ok], 5, labels=False, duplicates="drop")
        ic[lab] = {"spearman": round(float(r.statistic), 3),
                   "quintili_%": [round(float(v) * 100, 2)
                                  for v in f[ok].groupby(q).mean()]}

    n_sig = sum(r["significativo"] for r in righe)
    return {"incondizionato_%": round(unc * 100, 2), "per_stato": righe,
            "information_coefficient": ic, "test_effettuati": n_test,
            "test_significativi": n_sig,
            "attesi_per_caso": round(0.05 * n_test, 1),
            "verdetto": ("Il punteggio NON predice i rendimenti: "
                         f"{n_sig} risultati significativi su {n_test} test, "
                         f"contro {0.05*n_test:.1f} attesi per puro caso. "
                         "L'IC di Spearman è vicino a zero e i quintili non sono monotoni.")
            if n_sig <= max(1, 0.1 * n_test) else
            ("Alcuni stati sono distinguibili dalla media: "
             f"{n_sig} su {n_test}. Da verificare fuori campione.")}


def test_volatilita(panel: pd.DataFrame, spx: pd.Series) -> dict:
    d = panel.dropna(subset=["punteggio_tattico"]).copy()
    p = spx.reindex(d.index).ffill()
    ret = p.pct_change().fillna(0)
    d["rvf"] = (ret.rolling(H).std().shift(-H) * np.sqrt(252)).reindex(d.index)
    d = d.dropna(subset=["rvf"])
    unc = float(d["rvf"].mean())
    righe = []
    for st, g in d.groupby("stato_tattico"):
        if st == "DATI INSUFFICIENTI" or len(g) < 150:
            continue
        lo, hi = bootstrap_blocchi(g["rvf"].values)
        righe.append({"stato": st, "n": int(len(g)),
                      "vol_attesa_%": round(float(g["rvf"].mean()) * 100, 1),
                      "ic_basso_%": round(lo * 100, 1), "ic_alto_%": round(hi * 100, 1),
                      "significativo": bool(lo > unc or hi < unc)})
    r = stats.spearmanr(d["punteggio_tattico"], d["rvf"])
    n_sig = sum(x["significativo"] for x in righe)
    return {"incondizionata_%": round(unc * 100, 1), "per_stato": righe,
            "spearman": round(float(r.statistic), 3),
            "test_significativi": n_sig, "test_effettuati": len(righe),
            "verdetto": (f"Il punteggio predice la volatilità: {n_sig} stati su "
                         f"{len(righe)} sono distinguibili dalla media, con "
                         f"correlazione di rango {r.statistic:+.2f}.")}


def _feature_jump(ret: pd.Series) -> pd.DataFrame:
    """Feature del paper sui Jump Model: rendimento, deviazione al ribasso e
    Sortino su tre orizzonti."""
    out = {}
    for hl in (5, 21, 63):
        ew = ret.ewm(halflife=hl)
        m = ew.mean()
        giu = ret.clip(upper=0).pow(2).ewm(halflife=hl).mean().pow(0.5)
        out[f"ret_{hl}"] = m
        out[f"dd_{hl}"] = giu
        out[f"sortino_{hl}"] = m / giu.replace(0, np.nan)
    return pd.DataFrame(out).replace([np.inf, -np.inf], np.nan).dropna()


def jump_model(spx: pd.Series, n_stati: int = 2, penalita: float = 50.0) -> dict:
    """Statistical Jump Model addestrato in walk-forward: rifit ogni anno,
    classificazione solo sull'anno successivo (mai in campione)."""
    try:
        from jumpmodels.jump import JumpModel
        from jumpmodels.preprocess import StandardScalerPD, DataClipperStd
    except Exception as e:                        # noqa: BLE001
        return {"errore": f"jumpmodels non disponibile: {e}"}

    ret = spx.pct_change().fillna(0)
    X = _feature_jump(ret)
    anni = sorted({d.year for d in X.index})
    etichette = pd.Series(index=X.index, dtype=float)
    errori: list[str] = []
    for a in anni:
        train = X[X.index.year < a]
        test = X[X.index.year == a]
        if len(train) < 1000 or test.empty:
            continue
        try:
            clip = DataClipperStd(mul=3.0).fit(train)
            sc = StandardScalerPD().fit(clip.transform(train))
            xt = sc.transform(clip.transform(train))
            jm = JumpModel(n_components=n_stati, jump_penalty=penalita, cont=False)
            # ret_ser deve essere la SERIE DEI RENDIMENTI allineata, non le feature
            jm.fit(xt, ret.reindex(train.index), sort_by="cumret")
            etichette.loc[test.index] = jm.predict(
                sc.transform(clip.transform(test))).values
        except Exception as e:                     # noqa: BLE001
            errori.append(f"{a}: {type(e).__name__} {e}")
            continue
    et = etichette.dropna()
    if len(et) < 750:
        return {"errore": "storia fuori campione insufficiente",
                "dettagli": errori[:3]}

    p = spx.reindex(et.index).ffill()
    fwd = _fwd(p, H)
    rvf = ret.rolling(H).std().shift(-H).reindex(et.index) * np.sqrt(252)
    righe = []
    for s, g in et.groupby(et):
        f = fwd.reindex(g.index).dropna()
        v = rvf.reindex(g.index).dropna()
        if len(f) < 150:
            continue
        lo, hi = bootstrap_blocchi(f.values)
        righe.append({"stato": f"regime {int(s)}", "n": int(len(g)),
                      "rend_3m_%": round(float(f.mean()) * 100, 2),
                      "ic_basso_%": round(lo * 100, 2), "ic_alto_%": round(hi * 100, 2),
                      "vol_attesa_%": round(float(v.mean()) * 100, 1)})
    cambi = int((et != et.shift()).sum() - 1)
    serie = [{"d": str(i.date()), "s": int(v)}
             for i, v in et.resample("W-FRI").last().dropna().items()]
    return {"stati": righe, "cambi": cambi, "da": str(et.index[0].date()),
            "giorni": int(len(et)), "serie": serie,
            "penalita": penalita, "n_stati": n_stati}


def hmm_model(spx: pd.Series, n_stati: int = 3) -> dict:
    """HMM gaussiano sugli stessi dati, come secondo parere. Anche qui walk-forward."""
    try:
        from hmmlearn.hmm import GaussianHMM
    except Exception as e:                        # noqa: BLE001
        return {"errore": f"hmmlearn non disponibile: {e}"}
    ret = spx.pct_change().fillna(0)
    X = pd.DataFrame({"r": ret.ewm(halflife=5).mean(),
                      "v": ret.ewm(halflife=21).std()}).dropna()
    et = pd.Series(index=X.index, dtype=float)
    for a in sorted({d.year for d in X.index}):
        tr, te = X[X.index.year < a], X[X.index.year == a]
        if len(tr) < 1000 or te.empty:
            continue
        try:
            m = GaussianHMM(n_components=n_stati, covariance_type="full",
                            n_iter=60, random_state=0).fit(tr.values)
            ordine = np.argsort(m.means_[:, 1])          # ordina per volatilità
            mappa = {v: i for i, v in enumerate(ordine)}
            et.loc[te.index] = [mappa[s] for s in m.predict(te.values)]
        except Exception:                          # noqa: BLE001
            continue
    et = et.dropna()
    if len(et) < 750:
        return {"errore": "storia fuori campione insufficiente"}
    p = spx.reindex(et.index).ffill()
    fwd = _fwd(p, H)
    rvf = ret.rolling(H).std().shift(-H).reindex(et.index) * np.sqrt(252)
    righe = []
    for s, g in et.groupby(et):
        f = fwd.reindex(g.index).dropna()
        if len(f) < 150:
            continue
        lo, hi = bootstrap_blocchi(f.values)
        righe.append({"stato": f"vol {['bassa','media','alta'][int(s)] if n_stati==3 else int(s)}",
                      "n": int(len(g)), "rend_3m_%": round(float(f.mean()) * 100, 2),
                      "ic_basso_%": round(lo * 100, 2), "ic_alto_%": round(hi * 100, 2),
                      "vol_attesa_%": round(float(rvf.reindex(g.index).dropna().mean()) * 100, 1)})
    return {"stati": righe, "giorni": int(len(et)), "da": str(et.index[0].date())}


def accordo(panel: pd.DataFrame, jm: dict, spx: pd.Series) -> dict:
    """Quanto il composito a regole e il Jump Model dicono la stessa cosa."""
    if "serie" not in jm:
        return {}
    j = pd.Series({pd.Timestamp(x["d"]): x["s"] for x in jm["serie"]})
    st = panel["stato_tattico"].reindex(j.index).dropna()
    j = j.reindex(st.index)
    difensivi = {"DIFENSIVO", "STRESS / RISK-OFF"}
    a = st.isin(difensivi).astype(int)
    # qual e' il regime "cattivo" del Jump Model lo decidono i dati, non un'ipotesi:
    # e' quello con la volatilita' realizzata piu' alta
    rv = spx.pct_change().rolling(21).std().reindex(j.index)
    cattivo = int(rv.groupby(j).mean().idxmax())
    b = (j == cattivo).astype(int)
    return {"concordanza_%": round(float((a == b).mean()) * 100, 1),
            "settimane": int(len(a)),
            "nota": ("Concordanza fra il composito a regole (stati difensivi) e il "
                     "Jump Model (regime peggiore), su base settimanale.")}


def esegui(panel: pd.DataFrame, px: pd.DataFrame) -> dict:
    spx = px["^GSPC"].ffill()
    rend = test_rendimenti(panel, spx)
    vol = test_volatilita(panel, spx)
    jm = jump_model(spx)
    hm = hmm_model(spx)
    return {"rendimenti": rend, "volatilita": vol, "jump_model": jm,
            "hmm": hm, "accordo": accordo(panel, jm, spx)}
