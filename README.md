یه نسخه‌ی آماده‌ی README برات می‌نویسم که مستقیم بذاری تو ریپو و بعداً هرجا لازم شد جزئیات اضافه کنی.

***

```markdown
# IKCO Time Series Preprocessing & Classical Models

This project walks through **end-to-end preprocessing** and **classical time series modeling** on daily price data for the Tehran Stock Exchange stock *Iran Khodro (IKCO)* using Python, Pandas, NumPy, Matplotlib, and Statsmodels. [web:73][web:111]

The focus is on:
- clean, reproducible **data preprocessing** for financial time series
- building and interpreting **AR, MA, ARMA/ARIMA** models on returns

---

## 1. Data

- Source: CSV export of IKCO daily prices from the Tehran Stock Exchange (TSETMC / TseClient-style format). [web:25][web:88]
- Original columns (example):  
  `Unnamed: 0, <TICKER>, <DTYYYYMMDD>, <HIGH>, <LOW>, <CLOSE>, <VALUE>, <VOL>, <OPENINT>, <PER>, <OPEN>, <LAST>`  
- Target: clean **daily OHLCV** time series with a `DatetimeIndex` and additional engineered features.

The raw file used in the notebook is:

```text
Stock_IKCO1.csv
```

(You can replace it with any similar TSETMC-style CSV.)

---

## 2. Environment & Setup

Create a dedicated virtual environment (example: Windows + `venv`):

```bash
cd C:\Users\PC\Desktop\IKCO_preprocessing
python -m venv .venv
.\.venv\Scripts\activate  # or source .venv/bin/activate on Linux/Mac
pip install -r requirements.txt
```

Key dependencies:

- `numpy`
- `pandas`
- `matplotlib`
- `statsmodels` (for AR/MA/ARMA/ARIMA models) [web:111][web:118]

---

## 3. Project Structure

```text
IKCO_preprocessing/
├── Stock_IKCO1.csv          # raw IKCO daily prices (TSETMC format)
├── preprocessing.ipynb      # main EDA + preprocessing notebook
├── models_arima.ipynb       # AR/MA/ARMA/ARIMA on returns
├── requirements.txt
└── .venv/                   # local virtual environment (ignored in Git)
```

You can merge `preprocessing` and `models` into a single notebook if preferred.

---

## 4. Preprocessing Pipeline

All steps are implemented in the notebook using **Pandas/NumPy**. [web:73][web:65]

### 4.1 Load & clean raw data

1. **Read CSV**

```python
data = pd.read_csv("Stock_IKCO1.csv")
```

2. **Drop redundant index column**

```python
data = data.drop(columns=["Unnamed: 0"])
```

3. **Rename columns to simple names**

```python
data = data.rename(columns={
    "<TICKER>": "ticker",
    "<DTYYYYMMDD>": "date",
    "<HIGH>": "high",
    "<LOW>": "low",
    "<CLOSE>": "close",
    "<VALUE>": "value",
    "<VOL>": "volume",
    "<OPENINT>": "openint",
    "<PER>": "per",
    "<OPEN>": "open",
    "<LAST>": "last"
})
```

4. **Convert `date` to `DatetimeIndex`**

```python
data["date"] = pd.to_datetime(data["date"].astype(str), format="%Y%m%d")
df = data.set_index("date").sort_index()
```

This gives a properly ordered `DatetimeIndex`, which is crucial for resampling, rolling windows, and time-based slicing. [web:73][web:79]

---

### 4.2 Basic cleaning & sanity checks

1. **Ensure numeric types**

```python
cols = ["open", "high", "low", "close", "volume"]
df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
```

2. **Remove obviously invalid rows**

```python
mask_valid = (
    (df["open"] > 0) &
    (df["high"] > 0) &
    (df["low"]  > 0) &
    (df["close"] > 0) &
    (df["volume"] >= 0)
)
df = df[mask_valid]
```

3. **Check missing values & duplicates**

```python
df.isna().sum()
dup_dates = df.index.duplicated().sum()
```

(If needed, duplicate dates are removed using `~df.index.duplicated(keep="last")`.)

---

### 4.3 Feature engineering on daily data

Using **returns** rather than raw prices to approximate a stationary series. [file:110][web:73]

1. **Simple and log returns**

```python
df["ret"]      = df["close"].pct_change()
df["log_ret"]  = np.log(df["close"] / df["close"].shift(1))
df = df.dropna(subset=["ret"])
```

2. **Rolling statistics**

```python
df["ma_5"]   = df["close"].rolling(window=5).mean()
df["ma_20"]  = df["close"].rolling(window=20).mean()
df["vol_20"] = df["ret"].rolling(window=20).std()
```

3. **Outlier handling (example: clipping returns)**

```python
df["ret_clipped"]     = df["ret"].clip(-0.10, 0.10)
df["log_ret_clipped"] = df["log_ret"].clip(-0.10, 0.10)
```

4. **Weekly resampling (weekly OHLCV candles)**

```python
weekly = df.resample("W").agg({
    "open": "first",
    "high": "max",
    "low":  "min",
    "close": "last",
    "volume": "sum"
}).dropna(subset=["open", "high", "low", "close"])
```

---

## 5. Classical Time Series Models (Statsmodels)

Modeling is done on the **daily return series** `ret` using `statsmodels.tsa.arima.model.ARIMA`. [web:111][web:118]

### 5.1 ACF & PACF diagnostics

To get an idea of AR/MA orders: [file:110][web:119]

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

series = df["ret"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(series, lags=30, ax=axes)
axes.set_title("ACF of IKCO returns")

plot_pacf(series, lags=30, ax=axes)
axes.set_title("PACF of IKCO returns")
plt.tight_layout()
plt.show()
```

- PACF: used to identify AR order \(p\).  
- ACF: used to identify MA order \(q\). [file:110]

---

### 5.2 AR(p) on returns (ARIMA(p,0,0))

Example: AR(1): [file:110][web:111]

```python
from statsmodels.tsa.arima.model import ARIMA

model_ar1 = ARIMA(series, order=(1, 0, 0))
res_ar1 = model_ar1.fit()
print(res_ar1.summary())
```

Interpretation:
- `ar.L1` ≈ \(\rho_1\) → strength of dependence on previous day’s return.
- Residual ACF is checked to ensure no remaining autocorrelation.

---

### 5.3 MA(q) on returns (ARIMA(0,0,q))

Example: MA(1): [file:110][web:111]

```python
model_ma1 = ARIMA(series, order=(0, 0, 1))
res_ma1 = model_ma1.fit()
print(res_ma1.summary())
```

`ma.L1` captures dependence on the previous forecast error (shock).

---

### 5.4 ARMA(1,1) on returns (ARIMA(1,0,1))

Main working model in this project: [file:110][web:111]

```python
model_arma11 = ARIMA(series, order=(1, 0, 1))
res_arma11 = model_arma11.fit()
print(res_arma11.summary())
```

Example interpretation (based on IKCO results):

- `const` ≈ 0.0007 (not statistically significant) ⇒ mean daily return ≈ 0. [web:111]  
- `ar.L1` ≈ 0.39 (significant) ⇒ return_t has notable dependence on return_{t-1}. [file:110]  
- `ma.L1` ≈ −0.24 (significant) ⇒ the model also needs a moving-average term to capture structure in shocks. [file:110]  
- AIC ≈ −19199, and Ljung–Box test on residuals indicates no strong remaining autocorrelation at lag 1. [web:111][web:119]

Residual diagnostics:

```python
resid_arma = res_arma11.resid
plot_acf(resid_arma, lags=30)
plt.title("ACF of ARMA(1,1) residuals")
plt.show()
```

---

### 5.5 ARIMA on prices (sketch)

On raw prices `close`, differencing is needed to remove non-stationary trend/random walk: [file:110][web:111]

```python
price = df["close"]

model_arima = ARIMA(price, order=(1, 1, 1))  # ARIMA(p=1, d=1, q=1)
res_arima = model_arima.fit()
print(res_arima.summary())
```

`d=1` corresponds to modeling first differences of price (i.e., approximate returns).

---

## 6. Possible Extensions

Future directions (still in time series / quant context): [file:110][web:117]

- **GARCH(1,1)** on ARMA residuals to model conditional volatility.  
- **VAR** models on multiple stocks or indices.  
- **Granger causality** tests between IKCO and sector index returns.  
- Exporting cleaned features (`ret`, `log_ret`, moving averages, volatility, weekly candles) as input to ML or RL trading strategies.

---

## 7. How to Run

1. Clone the repo and create the virtual environment:

```bash
git clone <your-repo-url>.git
cd IKCO_preprocessing
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2. Open the project folder in VS Code and select the `.venv` Python interpreter. [web:50][web:51]  

3. Run the notebooks in order:

- `preprocessing.ipynb` → load & clean IKCO data, engineer features.  
- `models_arima.ipynb` → ACF/PACF, AR/MA/ARMA/ARIMA models on returns.  

4. (Optional) Replace `Stock_IKCO1.csv` with another TSETMC-style stock file and re-run all cells to reproduce the full pipeline for a different symbol.

---
```
