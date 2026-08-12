# Market Regime Cockpit

Dashboard di regime di mercato che si aggiorna da sola ogni mattina. Risponde a una
domanda sola: **in che regime siamo, e cosa è storicamente successo nei regimi come questo.**

Non prevede i prezzi. Descrive le condizioni e misura le frequenze condizionate.

---

## Come è fatto

Tre pezzi separati, ed è la separazione a renderlo affidabile:

| Pezzo | Cosa fa | Dove gira |
|---|---|---|
| **Motore** | Scarica i dati, ricalcola i regimi, scrive `out/payload.json` e `out/index.html` | GitHub Actions, ogni mattina alle 05:30 UTC nei giorni feriali |
| **Dashboard statica** | Pagina HTML autoconsistente, con tutte le serie dentro | GitHub Pages |
| **App interattiva** | Legge gli stessi dati già calcolati, aggiunge filtri e finestre temporali | Streamlit Cloud |

L'app non scarica niente al volo: apre in due secondi e non può essere bloccata dai
limiti di Yahoo. Ogni run committa i dati aggiornati nel repo, quindi **lo storico si
accumula da solo**, giorno dopo giorno.


---

## Il risultato che decide tutto il resto

Prima di credere a un modello di regime bisogna misurare se dice qualcosa di vero.
Misurato, con bootstrap a blocchi che tiene conto della sovrapposizione dei rendimenti
forward, sull'S&P 500 dal 1997:

**✗ Il punteggio NON predice i rendimenti.** IC di Spearman −0,06 a 1 mese, −0,04 a 3 mesi,
+0,02 a 6 mesi. I quintili non sono monotoni. Zero stati su cinque distinguibili dalla
media incondizionata. Un classificatore statistico indipendente (Statistical Jump Model,
walk-forward) arriva alla stessa conclusione.

**✓ Il punteggio PREDICE la volatilità.** Monotono e con ampiezza enorme: da 8,9% attesa a
3 mesi nel regime migliore a 22,7% nel peggiore. Quattro stati su cinque significativi.
Correlazione di rango −0,49.

> **Il regime non ti dice dove vanno i prezzi. Ti dice quanto rischio stai correndo per prenderli.**

Da qui discende l'unico uso difendibile: dimensionare l'esposizione inversamente alla
volatilità attesa. `esposizione = 12% / vol attesa`, tetto 1,5×, banda di non-intervento
di 5 punti, 5 bp di costi. Backtest 1999-2026 con previsione stimata a finestra espansiva:

| | CAGR | Vol | Sharpe | Perdita max | Sortino |
|---|---|---|---|---|---|
| Buy & hold S&P 500 | 5,13% | 16,9% | 0,30 | **−56,8%** | 0,34 |
| Solo vol realizzata (nessun regime) | 4,80% | 12,6% | 0,38 | −37,6% | 0,44 |
| Solo regime | 4,60% | 12,0% | 0,38 | −38,1% | 0,44 |
| **Regime + vol realizzata** | 4,79% | 11,8% | **0,41** | **−34,8%** | **0,48** |

Tre cose oneste. Il CAGR è **più basso** del buy & hold: è uno strumento di riduzione del
rischio, non di rendimento. Il regime **da solo perde** contro una banale volatilità
realizzata a 63 giorni (correlazione con la vol futura 0,37 contro 0,53). Serve **in
combinazione**, dove migliora la previsione (RMSE da 8,4 a 7,5 punti) e taglia il turnover.
Nel 2008-09 la regola perdeva 29% invece di 54%, e batte il drawdown del buy & hold in
tutti e sette i sottoperiodi testati.

---

## Come si legge la dashboard

L'ordine è deliberato: **la decisione in cima, i dati in fondo.**

1. **Livelli di rottura** — a che valore ogni variabile, da sola, farebbe cambiare stato al
   modello. Problema inverso risolto per bisezione rifacendo il conto completo. Le variabili
   che si muovono insieme (spread HY e CCC, VIX e VIX3M) vengono spostate insieme.
2. **Cosa è cambiato** — diff rispetto a ieri: pilastri, soglie attraversate, estremi toccati.
3. **Esposizione consigliata** con il suo track record misurato.
4. **Cosa il modello sa e non sa fare** — i test di significatività, in chiaro.
5. Grafici TradingView, pilastri, quadrante macro, storia.
6. Tabelle di riferimento, chiuse di default.

---

## Il modello

### Due letture, deliberatamente separate

**Tattica (1-4 settimane).** Media pesata di quattro pilastri:

| Pilastro | Peso | Cosa misura |
|---|---|---|
| Tendenza | 30% | prezzo contro medie, pendenza, drawdown, ampiezza settoriale e globale |
| Volatilità | 25% | VIX e struttura a termine, MOVE, vol realizzata, premio implicita-realizzata, asimmetria |
| Credito e liquidità | 25% | spread HY/IG/CCC, rapporti HYG-LQD-IEF, NFCI, stress index, liquidità netta della Fed |
| Propensione al rischio | 20% | azioni contro bond, biotech, growth-value, carry AUD/JPY, bitcoin, oro |

**Allocativa (1-6 mesi).** Quadrante crescita × inflazione, molto più lento:
Goldilocks, Reflazione, Stagflazione, Contrazione. Costruito su segnali di mercato
(rame/oro, ciclici/difensivi, curva, breakeven, commodity) più i dati macro veri.

Quando le due letture divergono, la divergenza *è* il segnale.

### Le tre scelte che contano

**1. I pesi non sono ottimizzati.** Sono scelti a priori. Ottimizzarli sul passato
produrrebbe un modello che spiega benissimo la storia e nulla del futuro.

**2. Isteresi sulle soglie.** Senza, il segnale cambiava stato ogni 9 giorni: inutile.
Con un margine di 6 punti la durata mediana di un regime passa a ~41 giorni e i
cambi dal 1997 scendono da 629 a 167.

**3. Sotto-blocchi tematici a peso uguale.** Dentro ogni pilastro i temi pesano uguale, non
le singole feature: senza, quattro spread di credito venivano diluiti da tredici rapporti
fra ETF e il pilastro era quasi insensibile all'HY OAS.

### Nessun look-ahead

- Gli z-score sono rolling a 3 anni, guardano solo indietro.
- Le serie macro sono **ritardate del reale tempo di pubblicazione**: il CPI di giugno
  entra nel modello a metà luglio, non il 1° giugno. Senza questa correzione il
  backtest userebbe informazione dal futuro e i risultati sarebbero fasulli.
- I rendimenti sono da *t* in avanti, il regime da *t* all'indietro.

---

## Dati

- **Yahoo Finance** — 88 strumenti: indici USA e globali, 15 settori, tassi, volatilità,
  credito, valute, materie prime, cripto. Nessuna chiave, storico dal 1990.
- **FRED** (Federal Reserve Bank of St. Louis) — 33 serie: spread OAS, condizioni
  finanziarie, bilancio Fed, curva, breakeven, occupazione, inflazione, immobiliare.
  Chiave gratuita su <https://fred.stlouisfed.org/docs/api/api_key.html>.

---

## Installazione

```bash
git clone https://github.com/Sbirrondi/market-regime-cockpit
cd market-regime-cockpit
pip install -r requirements.txt

export FRED_API_KEY=...        # la tua chiave
python run.py                  # scarica, calcola, genera out/index.html

streamlit run streamlit_app.py # l'app interattiva
```

Il primo run scarica ~30 anni di storico per 88 strumenti: mettici 5-8 minuti.
Quelli successivi sono incrementali e ci mettono meno di un minuto.

### Configurazione su GitHub

1. **Settings → Secrets and variables → Actions → New repository secret**:
   `FRED_API_KEY`
2. **Settings → Pages → Source: GitHub Actions**
3. **Actions → Aggiorna Market Regime Cockpit → Run workflow** per il primo giro

### Configurazione su Streamlit Cloud

Punta l'app a `streamlit_app.py` di questo repo. Nessun secret necessario: legge
i dati già committati.

---

## Struttura

```
run.py                      punto di ingresso
streamlit_app.py            vetrina interattiva
src/regime_cockpit/
  universe.py               gli 88 strumenti e i 17 rapporti cross-asset
  data.py                   fetch Yahoo e FRED, cache, ritardi di pubblicazione
  indicators.py             feature e punteggi dei pilastri
  regime.py                 classificazione, isteresi, direzione
  backtest.py               base rate condizionate, transizioni, episodi storici
  export.py                 costruzione del payload JSON
  template.html             la dashboard
data/cache/                 pannelli storici (parquet) — versionati, crescono nel tempo
out/                        payload.json e index.html pubblicati
```

---

## Difetti trovati e corretti

Vale la pena elencarli, perché sono il genere di errore che rende inutile un modello del genere
senza che nessuno se ne accorga.

- **Il pilastro volatilità era quasi cieco al VIX.** Il "premio di volatilità implicita"
  (VIX meno vol realizzata) esplode durante i crolli e, con segno positivo, annullava il
  segnale del VIX. Rimosso: il pilastro è passato da +17 a +6 sugli stessi dati.
- **Look-ahead sulle serie macro.** Erano datate al periodo di riferimento, non alla
  pubblicazione. Corretto con i ritardi reali.
- **Forward-fill illimitato.** Trascinava serie morte dentro i calcoli. Ora `ffill(limit=5)`:
  dopo una settimana la feature esce invece di mentire.
- **Leadership settoriale.** Era rumore (t-statistiche tutte sotto 1) presentato con due
  decimali. Rimossa.
- **Diluizione dentro i pilastri.** Risolta coi sotto-blocchi tematici.

## Limiti

- Le base rate degli stati estremi poggiano su poche centinaia di giorni concentrati
  in pochi episodi. Sono indizi, non statistiche solide.
- Il modello è addestrato implicitamente su un periodo (1997-2026) che contiene
  tre crisi e un regime di tassi anomalo. Non c'è ragione di credere che il futuro
  gli somigli.
- Yahoo limita per indirizzo IP: il motore fa backoff esponenziale e salva in modo
  incrementale, ma un run può richiedere qualche minuto in più.
- I ritardi di pubblicazione macro sono approssimati con valori fissi. La versione
  rigorosa userebbe i dati *vintage* di ALFRED.
