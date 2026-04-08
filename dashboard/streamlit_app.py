from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from model_utils import get_model_bundle  # noqa: E402

st.set_page_config(page_title="Dashboard HR Attrition", layout="wide")

DATA_PATH = BASE_DIR / "data" / "employee_data.csv"
MODEL_PATH = BASE_DIR / "model" / "attrition_model.joblib"
METRICS_PATH = BASE_DIR / "model" / "model_metrics.json"


@st.cache_data
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model_bundle():
    bundle, metrics, model_status = get_model_bundle(
        model_path=MODEL_PATH,
        data_path=DATA_PATH,
        metrics_path=METRICS_PATH,
        persist_if_retrained=False,
    )
    return bundle, metrics, model_status


def prepare_labeled_data(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    labeled = df.dropna(subset=["Attrition"]).copy()
    labeled["Attrition"] = labeled["Attrition"].astype(int)

    labeled["AgeBand"] = pd.cut(
        labeled["Age"],
        bins=[17, 25, 35, 45, 60],
        labels=["18-25", "26-35", "36-45", "46-60"],
    )
    labeled["IncomeBand"] = pd.qcut(
        labeled["MonthlyIncome"],
        q=4,
        labels=["Q1 rendah", "Q2", "Q3", "Q4 tinggi"],
        duplicates="drop",
    )
    labeled["TenureBand"] = pd.cut(
        labeled["YearsAtCompany"],
        bins=[-1, 2, 5, 10, 40],
        labels=["<=2", "3-5", "6-10", ">10"],
    )

    feature_cols = bundle["features"]
    X = labeled[feature_cols].copy()
    labeled["risk_score"] = bundle["pipeline"].predict_proba(X)[:, 1]
    return labeled


def agg_attrition(dataframe: pd.DataFrame, col: str) -> pd.DataFrame:
    return (
        dataframe.groupby(col, observed=False)["Attrition"]
        .mean()
        .mul(100)
        .reset_index(name="AttritionRate")
    )


def make_bar(dataframe: pd.DataFrame, x: str, title: str):
    chart = px.bar(
        dataframe,
        x=x,
        y="AttritionRate",
        text="AttritionRate",
        title=title,
    )
    chart.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    chart.update_layout(yaxis_title="Attrition Rate (%)")
    return chart


def infer_default_value(series: pd.Series):
    non_null = series.dropna()
    if non_null.empty:
        return None

    if pd.api.types.is_numeric_dtype(non_null):
        return float(non_null.median())

    mode_vals = non_null.mode()
    if len(mode_vals) > 0:
        return mode_vals.iloc[0]
    return non_null.iloc[0]


def build_prediction_input(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    st.subheader("Prediksi Attrition Karyawan")
    st.caption("Isi data karyawan berikut untuk memprediksi kemungkinan attrition.")

    input_data = {}
    cols = st.columns(2)

    for i, feature in enumerate(feature_cols):
        column_ui = cols[i % 2]
        series = df[feature] if feature in df.columns else pd.Series(dtype="object")
        default_value = infer_default_value(series)

        with column_ui:
            if pd.api.types.is_numeric_dtype(series.dropna()):
                series_non_null = series.dropna()
                min_val = float(series_non_null.min()) if not series_non_null.empty else 0.0
                max_val = float(series_non_null.max()) if not series_non_null.empty else 100.0
                value = float(default_value) if default_value is not None else min_val

                if pd.api.types.is_integer_dtype(series.dropna()):
                    input_data[feature] = st.number_input(
                        feature,
                        min_value=int(min_val),
                        max_value=int(max_val),
                        value=int(round(value)),
                        step=1,
                    )
                else:
                    input_data[feature] = st.number_input(
                        feature,
                        min_value=min_val,
                        max_value=max_val,
                        value=value,
                    )
            else:
                options = sorted(series.dropna().astype(str).unique().tolist())
                if not options:
                    options = ["Unknown"]

                default_str = str(default_value) if default_value is not None else options[0]
                default_index = options.index(default_str) if default_str in options else 0

                input_data[feature] = st.selectbox(
                    feature,
                    options=options,
                    index=default_index,
                )

    return pd.DataFrame([input_data])


def show_prediction_result(bundle: dict, input_df: pd.DataFrame):
    if st.button("Predict Attrition", type="primary"):
        try:
            probability = float(bundle["pipeline"].predict_proba(input_df)[0][1])
            prediction = int(bundle["pipeline"].predict(input_df)[0])

            st.markdown("### Hasil Prediksi")
            c1, c2 = st.columns(2)
            c1.metric("Probabilitas Attrition", f"{probability * 100:.1f}%")
            c2.metric("Prediksi", "Attrition" if prediction == 1 else "No Attrition")

            if prediction == 1:
                st.error("Karyawan diprediksi berisiko keluar.")
            else:
                st.success("Karyawan diprediksi cenderung bertahan.")

            if probability >= 0.7:
                st.warning("Risiko tinggi. Perlu intervensi cepat dari HR.")
            elif probability >= 0.4:
                st.info("Risiko menengah. Perlu monitoring lanjutan.")
            else:
                st.info("Risiko relatif rendah.")

            st.dataframe(input_df, use_container_width=True)

        except Exception as e:
            st.error(f"Prediksi gagal dijalankan: {e}")


df = load_dataset()
bundle, metrics, model_status = load_model_bundle()
labeled = prepare_labeled_data(df, bundle)

st.title("Dashboard Attrition HR - Jaya Jaya Maju")
st.caption(
    "Dashboard ini menampilkan faktor-faktor yang berkaitan dengan attrition dan menyediakan form prediksi attrition karyawan."
)

if model_status == "retrained_from_dataset":
    st.info(
        "Model tersimpan tidak kompatibel dengan environment saat ini, sehingga dashboard otomatis melatih ulang model dari dataset agar tetap dapat berjalan."
    )

tab1, tab2 = st.tabs(["Dashboard", "Prediksi"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Attrition rate", f"{labeled['Attrition'].mean() * 100:.1f}%")
    c2.metric("Observasi berlabel", f"{len(labeled):,}")
    c3.metric("Label attrition kosong", f"{df['Attrition'].isna().sum():,}")
    c4.metric("ROC-AUC model", f"{metrics.get('roc_auc', 0):.3f}")

    st.subheader("Faktor utama yang ditampilkan")
    row1 = st.columns(3)
    for col_obj, field, title in [
        (row1[0], "OverTime", "Attrition by Overtime"),
        (row1[1], "BusinessTravel", "Attrition by Business Travel"),
        (row1[2], "AgeBand", "Attrition by Age Band"),
    ]:
        col_obj.plotly_chart(
            make_bar(agg_attrition(labeled, field), field, title),
            use_container_width=True,
        )

    row2 = st.columns(3)
    for col_obj, field, title in [
        (row2[0], "IncomeBand", "Attrition by Income Band"),
        (row2[1], "TenureBand", "Attrition by Tenure Band"),
        (row2[2], "EnvironmentSatisfaction", "Attrition by Environment Satisfaction"),
    ]:
        col_obj.plotly_chart(
            make_bar(agg_attrition(labeled, field), field, title),
            use_container_width=True,
        )

    jobrole = (
        labeled.groupby("JobRole")["Attrition"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .head(6)
        .reset_index(name="AttritionRate")
    )

    importance = pd.DataFrame(
        {
            "Feature": [
                "OverTime",
                "JobRole",
                "EnvironmentSatisfaction",
                "MaritalStatus",
                "JobSatisfaction",
                "EducationField",
                "YearsAtCompany",
                "JobLevel",
            ],
            "Importance": [0.1108, 0.0567, 0.0296, 0.0143, 0.0141, 0.0127, 0.0074, 0.0071],
        }
    ).sort_values("Importance")

    left, right = st.columns([1.1, 0.9])
    left.plotly_chart(
        make_bar(jobrole, "JobRole", "Top Job Role by Attrition"),
        use_container_width=True,
    )
    right.plotly_chart(
        px.bar(
            importance,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top Model Drivers",
        ),
        use_container_width=True,
    )

    st.subheader("Segmen prioritas intervensi")
    segments = (
        labeled.groupby(["JobRole", "OverTime", "IncomeBand"], observed=False)
        .agg(
            n=("Attrition", "size"),
            attrition_rate=("Attrition", "mean"),
            mean_risk=("risk_score", "mean"),
            median_income=("MonthlyIncome", "median"),
        )
        .reset_index()
    )
    segments = segments[segments["n"] >= 12].sort_values(
        ["attrition_rate", "mean_risk"],
        ascending=False,
    ).head(10)
    segments["attrition_rate"] = (segments["attrition_rate"] * 100).round(1)
    segments["mean_risk"] = (segments["mean_risk"] * 100).round(1)
    st.dataframe(segments, use_container_width=True)

    st.subheader("Catatan penggunaan")
    st.markdown(
        "- Dashboard ini fokus pada data yang memiliki label attrition.\n"
        "- Tabel segmen membantu HR menentukan kelompok karyawan yang perlu diprioritaskan untuk intervensi.\n"
        "- Fitur prediksi tersedia pada tab **Prediksi**."
    )

with tab2:
    input_df = build_prediction_input(df, bundle["features"])
    show_prediction_result(bundle, input_df)