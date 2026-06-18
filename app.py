import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import List, Tuple

ASSET_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_assets() -> Tuple:
    model = joblib.load(ASSET_DIR / "model_best.pkl")
    scaler = joblib.load(ASSET_DIR / "scaler.pkl")
    feature_names = joblib.load(ASSET_DIR / "feature_names.pkl")
    return model, scaler, feature_names


# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.replace(-999, np.nan, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.fillna(0, inplace=True)

    if "MONTH" not in df.columns:
        year_ref = df["YEAR"].fillna(2000).astype(int)
        doy_ref = df["DOY"].fillna(1).astype(int)
        dates = pd.to_datetime(
            year_ref.astype(str) + doy_ref.astype(str).str.zfill(3),
            format="%Y%j",
            errors="coerce",
        )
        df["MONTH"] = dates.dt.month.fillna(1).astype(int)

    rain = df["PRECTOTCORR"]

    df["RAIN_BINARY"] = (rain > 1.0).astype(int)
    df["RAIN_LAG1"] = rain.shift(1)
    df["RAIN_LAG3"] = rain.shift(3)
    df["RAIN_LAG7"] = rain.shift(7)
    df["RAIN_BINARY_LAG1"] = df["RAIN_BINARY"].shift(1)
    df["RAIN_BINARY_LAG3"] = df["RAIN_BINARY"].shift(3)
    df["RAIN_BINARY_LAG7"] = df["RAIN_BINARY"].shift(7)
    df["RAIN_ROLL3"] = rain.shift(1).rolling(3, min_periods=1).mean()
    df["RAIN_ROLL7"] = rain.shift(1).rolling(7, min_periods=1).mean()
    df["RAIN_ROLL14"] = rain.shift(1).rolling(14, min_periods=1).mean()
    df["RAIN_STD7"] = rain.shift(1).rolling(7, min_periods=1).std().fillna(0)
    df["RAIN_STD14"] = rain.shift(1).rolling(14, min_periods=1).std().fillna(0)
    df["RAIN_EXPANDING_MEAN"] = rain.shift(1).expanding(min_periods=1).mean()

    df["TEMP_CHANGE"] = df["T2M"].diff()
    df["TEMP_RANGE"] = df["T2M_MAX"] - df["T2M_MIN"]
    df["TEMP_ROLL7"] = df["T2M"].shift(1).rolling(7, min_periods=1).mean()
    df["TEMP_ROLL14"] = df["T2M"].shift(1).rolling(14, min_periods=1).mean()
    df["TEMP_VOL7"] = df["T2M"].shift(1).rolling(7, min_periods=1).std().fillna(0)
    df["TEMP_VOL14"] = df["T2M"].shift(1).rolling(14, min_periods=1).std().fillna(0)

    df["RH2M_LAG1"] = df["RH2M"].shift(1)
    df["RH2M_ROLL7"] = df["RH2M"].shift(1).rolling(7, min_periods=1).mean()
    df["RH2M_ROLL14"] = df["RH2M"].shift(1).rolling(14, min_periods=1).mean()
    df["RH2M_ANOM7"] = df["RH2M"] - df["RH2M_ROLL7"]
    df["RH2M_ANOM14"] = df["RH2M"] - df["RH2M_ROLL14"]
    df["RH2M_LAG3"] = df["RH2M"].shift(3)
    df["RH2M_LAG7"] = df["RH2M"].shift(7)
    df["RH2M_STD7"] = df["RH2M"].shift(1).rolling(7, min_periods=1).std().fillna(0)

    df["PRESS_CHANGE"] = df["PS"].diff()
    df["PRESS_ROLL7"] = df["PS"].shift(1).rolling(7, min_periods=1).mean()
    df["PRESS_ROLL14"] = df["PS"].shift(1).rolling(14, min_periods=1).mean()
    df["PRESS_VOL7"] = df["PS"].shift(1).rolling(7, min_periods=1).std().fillna(0)
    df["PRESS_VOL14"] = df["PS"].shift(1).rolling(14, min_periods=1).std().fillna(0)
    df["PRESS_DIFF_3"] = df["PS"] - df["PS"].shift(3)
    df["PRESS_ANOM"] = df["PS"] - df["PS"].shift(1).rolling(14, min_periods=1).mean()

    df["WIND_CHANGE"] = df["WS10M"].diff()
    df["WIND_ROLL7"] = df["WS10M"].shift(1).rolling(7, min_periods=1).mean()
    df["WIND_ROLL14"] = df["WS10M"].shift(1).rolling(14, min_periods=1).mean()
    df["WS10M_LAG1"] = df["WS10M"].shift(1)
    df["WS10M_LAG3"] = df["WS10M"].shift(3)
    df["WIND_ROLL3"] = df["WS10M"].shift(1).rolling(3, min_periods=1).mean()

    df["T2M_DIFF_DEW"] = df["T2M"] - df["T2MDEW"]
    df["DEW_HUM"] = df["T2MDEW"] * df["RH2M"] / 100.0
    df["DEW_CHANGE"] = df["T2MDEW"].diff().shift(1)

    df["WIND_RH"] = df["WS10M"] * df["RH2M"]
    df["TEMP_HUM"] = df["T2M"] * df["RH2M"]
    df["HEAT_INDEX"] = (
        -8.78469475556
        + 1.61139411 * df["T2M"]
        + 2.33854883889 * df["RH2M"]
        - 0.14611605 * df["T2M"] * df["RH2M"]
        - 0.012308094 * df["T2M"] ** 2
        - 0.0164248277778 * df["RH2M"] ** 2
        + 0.002211732 * df["T2M"] ** 2 * df["RH2M"]
        + 0.00072546 * df["T2M"] * df["RH2M"] ** 2
        - 0.000003582 * df["T2M"] ** 2 * df["RH2M"] ** 2
    )

    doy = df["DOY"]
    month = df["MONTH"]
    df["DAY_SIN"] = np.sin(2 * np.pi * doy / 365.25)
    df["DAY_COS"] = np.cos(2 * np.pi * doy / 365.25)
    df["MONTH_SIN"] = np.sin(2 * np.pi * month / 12)
    df["MONTH_COS"] = np.cos(2 * np.pi * month / 12)

    rain_bin = df["RAIN_BINARY"]
    df["RAIN_EXPANDING_STD"] = (
        df["PRECTOTCORR"].shift(1).expanding(min_periods=2).std().fillna(0)
    )

    rain_streak = []
    dry_streak = []
    rain_gap = []
    r_s = d_s = r_g = 0
    last_rain = -1

    for i, rb in enumerate(rain_bin):
        if rb == 1:
            r_s += 1
            d_s = 0
            last_rain = i
            r_g = 0
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

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.fillna(0, inplace=True)

    return df


def validate_columns(df: pd.DataFrame, required_cols: List[str]) -> List[str]:
    return [c for c in required_cols if c not in df.columns]


def predict_next_day(
    df: pd.DataFrame,
    model,
    scaler,
    feature_names: List[str],
) -> Tuple[float, pd.DataFrame]:
    df_fe = feature_engineering(df)
    last = df_fe.iloc[[-1]].copy()
    X = last.reindex(
    columns=scaler.feature_names_in_,
    fill_value=0
    )   
    # DEBUG
    st.write("Jumlah feature_names:", len(feature_names))
    st.write("Scaler Order:")
    st.write(list(scaler.feature_names_in_)[:10])

    st.write("Input Order:")
    st.write(list(X.columns)[:10])

    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "feature": scaler.feature_names_in_,
            "importance": model.feature_importances_
        })

    st.dataframe(
        importance_df.sort_values(
            "importance",
            ascending=False
        ).head(20)
    )

    if hasattr(scaler, "feature_names_in_"):
        st.write("Jumlah fitur scaler:", len(scaler.feature_names_in_))

        st.write("5 fitur pertama scaler:")
        st.write(list(scaler.feature_names_in_)[:5])

        st.write("5 fitur pertama input:")
        st.write(list(X.columns)[:5])

        missing = set(scaler.feature_names_in_) - set(X.columns)
        extra = set(X.columns) - set(scaler.feature_names_in_)

        st.write("Missing features:")
        st.write(list(missing))

        st.write("Extra features:")
        st.write(list(extra))
        st.write("Shape X:")
        st.write(X.shape)
    X_scaled = scaler.transform(X) 

    if hasattr(model, "predict_proba"):
        prob_rain = float(model.predict_proba(X_scaled)[0][1])
    else:
        prediction = model.predict(X_scaled)[0]
        prob_rain = 1.0 if prediction == 1 else 0.0

    return prob_rain, df_fe


def main() -> None:
    st.set_page_config(
        page_title="Prediksi Hujan Besok",
        page_icon="🌧️",
        layout="wide",
    )

    st.title("🌧️ Prediksi Hujan Besok")
    st.markdown("### Sistem Smart City menggunakan data NASA POWER dan Machine Learning")

    st.info(
        "Upload file CSV NASA POWER yang memiliki kolom: YEAR, DOY, T2M, T2M_MAX, T2M_MIN, RH2M, WS10M, WS10M_MAX, WD10M, CLRSKY_SFC_SW_DWN, T2MDEW, PRECTOTCORR, PS"
    )

    model, scaler, feature_names = load_assets()

    uploaded_file = st.file_uploader("📂 Upload CSV NASA POWER", type=["csv"])

    if uploaded_file is None:
        st.warning("Silakan unggah file CSV NASA POWER terlebih dahulu.")
        return

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as error:
        st.error(f"Gagal membaca file CSV: {error}")
        return

    st.success("✅ File berhasil diupload")
    st.write("Preview Data:")
    st.dataframe(df.head())

    required_columns = [
        "YEAR",
        "DOY",
        "T2M",
        "T2M_MAX",
        "T2M_MIN",
        "RH2M",
        "WS10M",
        "WS10M_MAX",
        "WD10M",
        "CLRSKY_SFC_SW_DWN",
        "T2MDEW",
        "PRECTOTCORR",
        "PS",
    ]

    missing_columns = validate_columns(df, required_columns)

    if missing_columns:
        st.error(f"Kolom berikut tidak ditemukan: {', '.join(missing_columns)}")
        return

    if len(df) < 14:
        st.error("Dataset minimal harus memiliki 14 baris data.")
        return

    if st.button("🔍 Prediksi Sekarang"):
        with st.spinner("Melakukan feature engineering dan prediksi..."):
            try:
                prob_rain, df_fe = predict_next_day(df, model, scaler, feature_names)
            except Exception as error:
                st.error(f"Terjadi error saat prediksi: {error}")
                return

        prob_no_rain = 1.0 - prob_rain

        st.divider()

        if prob_rain >= 0.5:
            st.success("🌧️ Besok Diprediksi Hujan")
        else:
            st.info("☀️ Besok Diprediksi Tidak Hujan")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("☔ Probabilitas Hujan", f"{prob_rain*100:.1f}%")
        with col2:
            st.metric("☀️ Probabilitas Tidak Hujan", f"{prob_no_rain*100:.1f}%")

        st.divider()
        st.subheader("📊 Kondisi Cuaca Terakhir")
        display_df = df.copy()

        display_df.replace(-999, np.nan, inplace=True)
        display_df.ffill(inplace=True)
        display_df.bfill(inplace=True)
        last_raw = df_fe.iloc[-1]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🌡️ Suhu", f"{last_raw['T2M']:.1f} °C")
            st.metric("💧 Kelembaban", f"{last_raw['RH2M']:.1f}%")
        with col2:
            st.metric("🌬️ Angin", f"{last_raw['WS10M']:.2f} m/s")
            st.metric("🌂 Curah Hujan", f"{last_raw['PRECTOTCORR']:.1f} mm")
        with col3:
            st.metric("🌫️ Titik Embun", f"{last_raw['T2MDEW']:.1f} °C")
            st.metric("⏲️ Tekanan", f"{last_raw['PS']:.2f} kPa")

        st.divider()
        st.write("**Fitur input terakhir setelah feature engineering:**")
        st.dataframe(df_fe.iloc[[-1]].T)


if __name__ == "__main__":
    main()
