import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import List, Tuple, Any

# ─────────────────────────────────────────────
# PENGATURAN AWAL
# ─────────────────────────────────────────────
ASSET_DIR = Path(__file__).resolve().parent
THRESHOLD_RAIN = 0.66

REQUIRED_COLUMNS = [
    "YEAR", "DOY", "T2M", "T2M_MAX", "T2M_MIN", "RH2M", 
    "WS10M", "WS10M_MAX", "WD10M", "CLRSKY_SFC_SW_DWN", 
    "T2MDEW", "PRECTOTCORR", "PS"
]

@st.cache_resource
def load_assets() -> Tuple[Any, Any]:
    """Memuat model dan scaler."""
    model = joblib.load(ASSET_DIR / "model_best.pkl")
    scaler = joblib.load(ASSET_DIR / "scaler.pkl")
    return model, scaler

# ─────────────────────────────────────────────
# LOGIKA PEMROSESAN DATA (TIDAK ADA ST.WRITE DI SINI)
# ─────────────────────────────────────────────
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Melakukan ekstraksi fitur dan transformasi data mentah."""
    df = df.copy()
    df.replace(-999, np.nan, inplace=True)
    df = df.ffill().bfill().fillna(0)

    if "MONTH" not in df.columns:
        year_ref = df["YEAR"].fillna(2000).astype(int)
        doy_ref = df["DOY"].fillna(1).astype(int)
        dates = pd.to_datetime(
            year_ref.astype(str) + doy_ref.astype(str).str.zfill(3),
            format="%Y%j", errors="coerce",
        )
        df["MONTH"] = dates.dt.month.fillna(1).astype(int)

    df["DAY_SIN"] = np.sin(2 * np.pi * df["DOY"] / 365.25)
    df["DAY_COS"] = np.cos(2 * np.pi * df["DOY"] / 365.25)
    df["MONTH_SIN"] = np.sin(2 * np.pi * df["MONTH"] / 12)
    df["MONTH_COS"] = np.cos(2 * np.pi * df["MONTH"] / 12)

    rain = df["PRECTOTCORR"]
    df["RAIN_BINARY"] = (rain > 1.0).astype(int)
    
    for lag in [1, 3, 7]:
        df[f"RAIN_LAG{lag}"] = rain.shift(lag)
        df[f"RAIN_BINARY_LAG{lag}"] = df["RAIN_BINARY"].shift(lag)
        
    for window in [3, 7, 14]:
        df[f"RAIN_ROLL{window}"] = rain.shift(1).rolling(window, min_periods=1).mean()
        
    df["RAIN_STD7"] = rain.shift(1).rolling(7, min_periods=1).std().fillna(0)
    df["RAIN_STD14"] = rain.shift(1).rolling(14, min_periods=1).std().fillna(0)
    df["RAIN_EXPANDING_MEAN"] = rain.shift(1).expanding(min_periods=1).mean()
    df["RAIN_EXPANDING_STD"] = rain.shift(1).expanding(min_periods=2).std().fillna(0)

    # 4. Kalkulasi Streak Hujan & Kering
    rain_streak, dry_streak, rain_gap = [], [], []
    r_s = d_s = r_g = 0  # <-- PERBAIKAN: r_g dideklarasikan di awal
    last_rain = -1

    for i, rb in enumerate(df["RAIN_BINARY"]):
        if rb == 1:
            r_s += 1
            d_s = 0
            r_g = 0      # <-- PERBAIKAN: r_g di-reset jadi 0 saat hujan
            last_rain = i
        else:
            d_s += 1
            r_s = 0
            r_g = i - last_rain if last_rain >= 0 else i + 1

        rain_streak.append(r_s)
        dry_streak.append(d_s)
        rain_gap.append(r_g)

    df["RAIN_STREAK"] = rain_streak
    df["DRY_STREAK"] = dry_streak
    df["RAIN_GAP"] = rain_gap


    df["TEMP_CHANGE"] = df["T2M"].diff()
    df["TEMP_RANGE"] = df["T2M_MAX"] - df["T2M_MIN"]
    for window in [7, 14]:
        df[f"TEMP_ROLL{window}"] = df["T2M"].shift(1).rolling(window, min_periods=1).mean()
        df[f"TEMP_VOL{window}"] = df["T2M"].shift(1).rolling(window, min_periods=1).std().fillna(0)

    for lag in [1, 3, 7]:
        df[f"RH2M_LAG{lag}"] = df["RH2M"].shift(lag)
    for window in [7, 14]:
        df[f"RH2M_ROLL{window}"] = df["RH2M"].shift(1).rolling(window, min_periods=1).mean()
        df[f"RH2M_ANOM{window}"] = df["RH2M"] - df[f"RH2M_ROLL{window}"]
    df["RH2M_STD7"] = df["RH2M"].shift(1).rolling(7, min_periods=1).std().fillna(0)

    df["PRESS_CHANGE"] = df["PS"].diff()
    df["PRESS_DIFF_3"] = df["PS"] - df["PS"].shift(3)
    df["PRESS_ANOM"] = df["PS"] - df["PS"].shift(1).rolling(14, min_periods=1).mean()
    for window in [7, 14]:
        df[f"PRESS_ROLL{window}"] = df["PS"].shift(1).rolling(window, min_periods=1).mean()
        df[f"PRESS_VOL{window}"] = df["PS"].shift(1).rolling(window, min_periods=1).std().fillna(0)

    df["WIND_CHANGE"] = df["WS10M"].diff()
    for lag in [1, 3]:
        df[f"WS10M_LAG{lag}"] = df["WS10M"].shift(lag)
    for window in [3, 7, 14]:
        df[f"WIND_ROLL{window}"] = df["WS10M"].shift(1).rolling(window, min_periods=1).mean()

    df["T2M_DIFF_DEW"] = df["T2M"] - df["T2MDEW"]
    df["DEW_HUM"] = df["T2MDEW"] * df["RH2M"] / 100.0
    df["DEW_CHANGE"] = df["T2MDEW"].diff().shift(1)
    df["WIND_RH"] = df["WS10M"] * df["RH2M"]
    df["TEMP_HUM"] = df["T2M"] * df["RH2M"]
    
    df["HEAT_INDEX"] = (
        -8.78469475556 + 1.61139411 * df["T2M"] + 2.33854883889 * df["RH2M"]
        - 0.14611605 * df["T2M"] * df["RH2M"] - 0.012308094 * df["T2M"] ** 2
        - 0.0164248277778 * df["RH2M"] ** 2 + 0.002211732 * df["T2M"] ** 2 * df["RH2M"]
        + 0.00072546 * df["T2M"] * df["RH2M"] ** 2 - 0.000003582 * df["T2M"] ** 2 * df["RH2M"] ** 2
    )

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df.ffill().bfill().fillna(0)

def predict_next_day(df: pd.DataFrame, model: Any, scaler: Any) -> float:
    """Melakukan prediksi probabilitas hujan tanpa mencetak log apapun."""
    df_fe = feature_engineering(df)
    last_row = df_fe.iloc[[-1]].copy()
    X = last_row.reindex(columns=scaler.feature_names_in_, fill_value=0)
    X_scaled = scaler.transform(X) 
    
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(X_scaled)[0][1])
    
    return 1.0 if model.predict(X_scaled)[0] == 1 else 0.0

# ─────────────────────────────────────────────
# ANTARMUKA PENGGUNA (UI) YANG BERSIH
# ─────────────────────────────────────────────
def main() -> None:
    st.set_page_config(page_title="Prediksi Cuaca", page_icon="🌤️", layout="centered")

    # Header
    st.markdown("<h1 style='text-align: center;'>🌤️ Prediksi Hujan Smart City</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Sistem Prediksi Cuaca Berbasis Machine Learning & Data NASA POWER</p>", unsafe_allow_html=True)
    st.write("---")

    # Load Assets
    try:
        model, scaler = load_assets()
    except Exception as e:
        st.error("⚠️ Sistem sedang dalam pemeliharaan (Gagal memuat model).")
        return

    # File Upload Widget
    uploaded_file = st.file_uploader("📂 Unggah File Data Cuaca (.csv)", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ File berhasil diunggah!")
        except Exception:
            st.error("⚠️ Gagal membaca file CSV. Pastikan format sudah benar.")
            return

        # Validasi Kolom
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            st.warning(f"⚠️ Data tidak lengkap. Kolom yang hilang: {', '.join(missing_cols)}")
            return

        if len(df) < 14:
            st.warning("⚠️ Membutuhkan minimal 14 hari data untuk melakukan prediksi yang akurat.")
            return

        # Sembunyikan tabel data di dalam expander agar layar tetap bersih
        with st.expander("Lihat sekilas data yang diunggah"):
            st.dataframe(df.tail(5))

        st.write("") # Spacing

        # Tombol Prediksi Utama
        if st.button("🔍 Analisis & Prediksi Cuaca Besok", type="primary", use_container_width=True):
            with st.spinner("Menganalisis pola cuaca..."):
                prob_rain = predict_next_day(df, model, scaler)
                prob_no_rain = 1.0 - prob_rain

            st.write("---")
            
            # --- TAMPILAN HASIL PREDIKSI ---
            if prob_rain >= THRESHOLD_RAIN:
                st.info("### 🌧️ Kesimpulan: Besok Diprediksi **HUJAN**")
            else:
                st.warning("### ☀️ Kesimpulan: Besok Diprediksi **CERAH** / TIDAK HUJAN")

            # Progress bar visual untuk persentase
            st.progress(prob_rain, text=f"Probabilitas Turun Hujan: {prob_rain*100:.1f}%")

            # --- TAMPILAN KONDISI CUACA HARI INI ---
            st.write("---")
            st.markdown("#### 📊 Kondisi Cuaca Hari Ini (Berdasarkan data terakhir)")
            
            last_raw = df.copy().replace(-999, np.nan).ffill().bfill().iloc[-1]
            
            # Menggunakan metrik bergaya dashboard
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="🌡️ Suhu Udara", value=f"{last_raw['T2M']:.1f} °C")
                st.metric(label="💧 Kelembaban", value=f"{last_raw['RH2M']:.0f}%")
            with col2:
                st.metric(label="🌬️ Kecepatan Angin", value=f"{last_raw['WS10M']:.1f} m/s")
                st.metric(label="🌂 Curah Hujan", value=f"{last_raw['PRECTOTCORR']:.1f} mm")
            with col3:
                st.metric(label="🌫️ Titik Embun", value=f"{last_raw['T2MDEW']:.1f} °C")
                st.metric(label="⏲️ Tekanan Udara", value=f"{last_raw['PS']:.1f} kPa")

if __name__ == "__main__":
    main()