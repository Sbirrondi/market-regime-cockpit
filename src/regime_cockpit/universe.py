"""Universo di strumenti monitorati dal Market Regime Cockpit.

Ogni voce: ticker Yahoo -> (nome leggibile, blocco, sottogruppo)
"""

# ---------------------------------------------------------------- USA CORE
US_INDICES = {
    "^GSPC": ("S&P 500", "equity_us", "index"),
    "^NDX": ("Nasdaq 100", "equity_us", "index"),
    "^DJI": ("Dow Jones", "equity_us", "index"),
    "^RUT": ("Russell 2000", "equity_us", "index"),
    "^SOX": ("Philadelphia Semi", "equity_us", "index"),
}

US_SECTORS = {
    "XLK": ("Technology", "sector", "cyclical_growth"),
    "XLF": ("Financials", "sector", "cyclical_value"),
    "XLE": ("Energy", "sector", "cyclical_value"),
    "XLV": ("Health Care", "sector", "defensive"),
    "XLI": ("Industrials", "sector", "cyclical_value"),
    "XLB": ("Materials", "sector", "cyclical_value"),
    "XLC": ("Communication", "sector", "cyclical_growth"),
    "XLRE": ("Real Estate", "sector", "rate_sensitive"),
    "XLY": ("Cons. Discretionary", "sector", "cyclical_growth"),
    "XLP": ("Cons. Staples", "sector", "defensive"),
    "XLU": ("Utilities", "sector", "defensive"),
    "SMH": ("Semiconductors", "sector", "cyclical_growth"),
    "XBI": ("Biotech", "sector", "high_beta"),
    "ITA": ("Aerospace & Defense", "sector", "cyclical_value"),
    "IGV": ("Software", "sector", "cyclical_growth"),
}

# ------------------------------------------------------- VOLATILITA / STRESS
VOLATILITY = {
    "^VIX": ("VIX (30d)", "vol", "equity_vol"),
    "^VIX3M": ("VIX 3 mesi", "vol", "equity_vol"),
    "^VIX9D": ("VIX 9 giorni", "vol", "equity_vol"),
    "^VVIX": ("VVIX (vol della vol)", "vol", "equity_vol"),
    "^MOVE": ("MOVE (vol bond)", "vol", "rates_vol"),
    "^OVX": ("OVX (vol oil)", "vol", "commodity_vol"),
    "^GVZ": ("GVZ (vol oro)", "vol", "commodity_vol"),
}

# --------------------------------------------------------------- TASSI USA
US_RATES = {
    "^IRX": ("T-Bill 13 sett.", "rates", "us_curve"),
    "^FVX": ("UST 5 anni", "rates", "us_curve"),
    "^TNX": ("UST 10 anni", "rates", "us_curve"),
    "^TYX": ("UST 30 anni", "rates", "us_curve"),
}

# --------------------------------------------- CREDITO / BOND (proxy ETF)
CREDIT_BONDS = {
    "HYG": ("HY Corporate", "credit", "high_yield"),
    "JNK": ("HY Corporate (JNK)", "credit", "high_yield"),
    "LQD": ("IG Corporate", "credit", "inv_grade"),
    "EMB": ("EM Sovereign USD", "credit", "em_debt"),
    "TLT": ("UST 20+ anni", "bonds", "duration"),
    "IEF": ("UST 7-10 anni", "bonds", "duration"),
    "SHY": ("UST 1-3 anni", "bonds", "short"),
    "TIP": ("TIPS", "bonds", "inflation_linked"),
    "BKLN": ("Senior Loans", "credit", "floating"),
}

# ------------------------------------------------------------ EQUITY GLOBAL
GLOBAL_EQUITY = {
    "^STOXX50E": ("Euro Stoxx 50", "equity_global", "europe"),
    "^GDAXI": ("DAX", "equity_global", "europe"),
    "^FTSE": ("FTSE 100", "equity_global", "europe"),
    "FTSEMIB.MI": ("FTSE MIB", "equity_global", "europe"),
    "^N225": ("Nikkei 225", "equity_global", "japan"),
    "^HSI": ("Hang Seng", "equity_global", "china"),
    "000001.SS": ("Shanghai Composite", "equity_global", "china"),
    "^KS11": ("KOSPI", "equity_global", "asia"),
    "^TWII": ("Taiwan Weighted", "equity_global", "asia"),
    "^BVSP": ("Bovespa", "equity_global", "latam"),
    "^NSEI": ("Nifty 50", "equity_global", "india"),
}

GLOBAL_ETF = {
    "EFA": ("MSCI EAFE", "equity_global", "dm_exus"),
    "EEM": ("MSCI EM", "equity_global", "em"),
    "FXI": ("China Large Cap", "equity_global", "china"),
    "KWEB": ("China Internet", "equity_global", "china"),
    "EWJ": ("Japan", "equity_global", "japan"),
    "EWG": ("Germany", "equity_global", "europe"),
    "EWU": ("UK", "equity_global", "europe"),
    "INDA": ("India", "equity_global", "india"),
    "EWZ": ("Brazil", "equity_global", "latam"),
    "VGK": ("Europe", "equity_global", "europe"),
}

# --------------------------------------------------------------------- FX
FX = {
    "DX-Y.NYB": ("Dollar Index (DXY)", "fx", "usd"),
    "EURUSD=X": ("EUR/USD", "fx", "major"),
    "USDJPY=X": ("USD/JPY", "fx", "major"),
    "GBPUSD=X": ("GBP/USD", "fx", "major"),
    "AUDJPY=X": ("AUD/JPY (risk proxy)", "fx", "risk_proxy"),
    "USDCNH=X": ("USD/CNH", "fx", "em"),
    "USDCHF=X": ("USD/CHF (safe haven)", "fx", "safe_haven"),
    "USDMXN=X": ("USD/MXN (EM carry)", "fx", "em"),
}

# ------------------------------------------------------------- COMMODITIES
COMMODITIES = {
    "GC=F": ("Oro", "commodity", "precious"),
    "SI=F": ("Argento", "commodity", "precious"),
    "HG=F": ("Rame", "commodity", "industrial"),
    "CL=F": ("WTI Crude", "commodity", "energy"),
    "BZ=F": ("Brent Crude", "commodity", "energy"),
    "NG=F": ("Natural Gas", "commodity", "energy"),
    "ZC=F": ("Mais", "commodity", "agri"),
    "DBC": ("Commodity Broad", "commodity", "broad"),
}

# ------------------------------------------------------------------ CRYPTO
CRYPTO = {
    "BTC-USD": ("Bitcoin", "crypto", "major"),
    "ETH-USD": ("Ethereum", "crypto", "major"),
    "SOL-USD": ("Solana", "crypto", "alt"),
}

# ---------------------------------------------- FATTORI / STILI (per ratio)
FACTORS = {
    "SPY": ("S&P 500 ETF", "factor", "beta"),
    "RSP": ("S&P Equal Weight", "factor", "breadth"),
    "IWM": ("Russell 2000 ETF", "factor", "size"),
    "IWF": ("Russell 1000 Growth", "factor", "style"),
    "IWD": ("Russell 1000 Value", "factor", "style"),
    "MTUM": ("Momentum", "factor", "style"),
    "USMV": ("Min Volatility", "factor", "style"),
    "QQQ": ("Nasdaq 100 ETF", "factor", "beta"),
}

UNIVERSE = {}
for _blk in (US_INDICES, US_SECTORS, VOLATILITY, US_RATES, CREDIT_BONDS,
             GLOBAL_EQUITY, GLOBAL_ETF, FX, COMMODITIES, CRYPTO, FACTORS):
    UNIVERSE.update(_blk)

TICKERS = list(UNIVERSE.keys())


# ------------------------------------------------------------------ RATIO
# Coppie il cui *rapporto* e' il segnale (num, den, nome, interpretazione)
# direction = +1 : rapporto in salita = RISK-ON
RATIOS = [
    ("RSP", "SPY", "Equal-weight / Cap-weight", "breadth", +1),
    ("IWM", "SPY", "Small cap / Large cap", "breadth", +1),
    ("XLY", "XLP", "Ciclici / Difensivi", "risk_appetite", +1),
    ("HYG", "LQD", "High Yield / Investment Grade", "credit", +1),
    ("HYG", "IEF", "High Yield / Treasury", "credit", +1),
    ("LQD", "IEF", "IG Credit / Treasury", "credit", +1),
    ("HG=F", "GC=F", "Rame / Oro", "growth", +1),
    ("SPY", "TLT", "Azioni / Bond lunghi", "risk_appetite", +1),
    ("IWF", "IWD", "Growth / Value", "style", 0),
    ("SMH", "SPY", "Semis / Mercato", "risk_appetite", +1),
    ("EEM", "SPY", "Emergenti / USA", "risk_appetite", +1),
    ("EFA", "SPY", "Internazionale / USA", "risk_appetite", 0),
    ("XBI", "XLV", "Biotech / Health Care", "risk_appetite", +1),
    ("TIP", "IEF", "TIPS / Nominali (breakeven proxy)", "inflation", 0),
    ("XLU", "SPY", "Utilities / Mercato", "defensive", -1),
    ("USMV", "SPY", "Min Vol / Mercato", "defensive", -1),
    ("GC=F", "SPY", "Oro / Azioni", "defensive", -1),
]
