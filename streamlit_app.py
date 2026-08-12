"""Vetrina interattiva su Streamlit Cloud.

Non scarica nulla: legge i dati che GitHub Actions ha gia' calcolato e
committato nel repo. Per questo apre in un paio di secondi e non puo'
essere bloccata dai limiti di Yahoo.
"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).parent
st.set_page_config(page_title="Market Regime Cockpit", page_icon="📊", layout="wide")


@st.cache_data(ttl=900)
def carica():
    payload = json.loads((BASE / "out" / "payload.json").read_text("utf8"))
    panel = pd.read_parquet(BASE / "data" / "cache" / "regime_panel.parquet")
    prezzi = pd.read_parquet(BASE / "data" / "cache" / "prices.parquet")
    return payload, panel, prezzi


COL = {"RISK-ON PIENO": "#0ca30c", "COSTRUTTIVO": "#0ca30c",
       "NEUTRALE / LATERALE": "#898781", "DIFENSIVO": "#ec835a",
       "STRESS / RISK-OFF": "#d03b3b", "DATI INSUFFICIENTI": "#898781"}

P, panel, prezzi = carica()
snap = P["snapshot"]
t = snap["tattico"]

st.title("Market Regime Cockpit")
st.caption(f"dati al {P['meta']['ultimo_dato']} · {P['meta']['strumenti']} strumenti · "
           f"{P['meta']['serie_macro']} serie macro · aggiornato {P['meta']['generato']}")

c1, c2, c3, c4 = st.columns([1.1, 1, 1, 1])
c1.metric("Punteggio tattico", f"{t['punteggio']:+.0f}", f"{t.get('var_21g', 0):+.1f} in 21 giorni")
c2.metric("Stato", t["stato"], t.get("direzione", ""))
if "allocativo" in snap:
    c3.metric("Quadrante macro", snap["allocativo"]["quadrante"],
              f"da {snap['allocativo']['giorni_nel_quadrante']} giorni")
if "fiducia" in snap:
    c4.metric("Consenso fra i pilastri", f"{snap['fiducia']['consenso']:.0f}%",
              f"{snap['fiducia']['pilastri_concordi']}/{snap['fiducia']['pilastri_totali']} d'accordo")

st.info(f"**{t['stato']} · {t.get('direzione','')}** — {t['descrizione']}"
        + (f"  \n**Sul piano allocativo:** {snap['allocativo']['descrizione']}"
           if "allocativo" in snap else ""))

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Storia del regime", "Pilastri", "Base rate", "Mercati", "Macro"])

with tab1:
    orizzonte = st.select_slider("Finestra", ["1 anno", "3 anni", "5 anni", "10 anni", "tutto"],
                                 value="5 anni")
    anni = {"1 anno": 1, "3 anni": 3, "5 anni": 5, "10 anni": 10, "tutto": 99}[orizzonte]
    d = panel.dropna(subset=["punteggio_tattico"])
    d = d[d.index >= d.index.max() - pd.DateOffset(years=anni)]
    spx = prezzi["^GSPC"].reindex(d.index).ffill()

    fig = go.Figure()
    for stato, col in COL.items():
        m = d[d["stato_tattico"] == stato]
        if len(m):
            fig.add_bar(x=m.index, y=m["punteggio_tattico"], name=stato,
                        marker_color=col, marker_line_width=0,
                        opacity=1.0 if stato != "COSTRUTTIVO" else .55)
    fig.update_layout(template="plotly_dark", height=300, barmode="relative",
                      margin=dict(l=0, r=0, t=10, b=0), bargap=0,
                      yaxis_title="punteggio tattico", legend_orientation="h")
    st.plotly_chart(fig, use_container_width=True)

    f2 = go.Figure(go.Scatter(x=d.index, y=spx, line=dict(color="#c3c2b7", width=2), name="S&P 500"))
    f2.update_layout(template="plotly_dark", height=240, margin=dict(l=0, r=0, t=10, b=0),
                     yaxis_type="log", yaxis_title="S&P 500 (log)")
    st.plotly_chart(f2, use_container_width=True)

with tab2:
    nomi = {"tendenza": "Tendenza", "volatilita": "Volatilità", "credito": "Credito e liquidità",
            "propensione_rischio": "Propensione al rischio", "crescita": "Crescita (macro)",
            "inflazione": "Inflazione (macro)"}
    for k, v in snap["pilastri"].items():
        if v is None:
            continue
        st.markdown(f"**{nomi.get(k, k)}** · `{v:+.0f}`")
        st.progress(min(1.0, max(0.0, (v + 100) / 200)))
        drv = P["pilastri_dettaglio"].get(k, [])
        if drv:
            with st.expander("cosa lo sta spingendo"):
                st.dataframe(pd.DataFrame(drv), hide_index=True, use_container_width=True)

with tab3:
    st.caption("Rendimenti forward dell'S&P 500 condizionati al regime. Non sono previsioni.")
    st.dataframe(pd.DataFrame(P["base_rate_2d"]), hide_index=True, use_container_width=True)
    st.caption("Episodi di allerta riconosciuti dal modello")
    st.dataframe(pd.DataFrame(P["episodi"]), hide_index=True, use_container_width=True)

with tab4:
    blocco = st.selectbox("Blocco", list(P["mercati"].keys()),
                          format_func=lambda s: s.replace("_", " ").capitalize())
    st.dataframe(pd.DataFrame(P["mercati"][blocco]), hide_index=True, use_container_width=True)
    st.caption("Rapporti cross-asset")
    st.dataframe(pd.DataFrame(P["rapporti"]), hide_index=True, use_container_width=True)

with tab5:
    st.dataframe(pd.DataFrame(P["macro"]), hide_index=True, use_container_width=True)
