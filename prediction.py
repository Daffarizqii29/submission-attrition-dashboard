import argparse
from pathlib import Path

import pandas as pd

from model_utils import DEFAULT_DATA_PATH, DEFAULT_METRICS_PATH, get_model_bundle


def main():
    parser = argparse.ArgumentParser(
        description="Prediksi risiko attrition karyawan dari file CSV (bisa untuk 1, 3, atau banyak karyawan)."
    )
    parser.add_argument("--input", required=True, help="Path CSV input")
    parser.add_argument("--output", default="prediksi_attrition.csv", help="Path CSV output")
    parser.add_argument("--model", default="model/attrition_model.joblib", help="Path file model")
    args = parser.parse_args()

    model_path = Path(args.model)
    bundle, metrics, model_status = get_model_bundle(
        model_path=model_path,
        data_path=DEFAULT_DATA_PATH,
        metrics_path=DEFAULT_METRICS_PATH,
        persist_if_retrained=False,
    )
    pipeline = bundle["pipeline"]
    threshold = bundle["threshold"]
    expected_features = bundle["features"]

    df = pd.read_csv(args.input)
    missing_cols = [c for c in expected_features if c not in df.columns]
    if missing_cols:
        raise ValueError(
            "Kolom input belum lengkap. Kolom yang hilang: " + ", ".join(missing_cols)
        )

    X = df[expected_features].copy()
    proba = pipeline.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    output = df.copy()
    output["attrition_risk_score"] = proba.round(4)
    output["predicted_attrition"] = pred
    output.to_csv(args.output, index=False)

    print(f"Selesai. Hasil prediksi disimpan ke {args.output}")
    print(f"Jumlah karyawan yang diproses: {len(df)}")
    print(f"Threshold model yang dipakai: {threshold:.2f}")
    if metrics:
        print(f"ROC-AUC referensi model: {metrics.get('roc_auc', 0):.3f}")
    if model_status == "retrained_from_dataset":
        print("Model bawaan tidak kompatibel dengan environment saat ini, sehingga model dilatih ulang otomatis dari dataset.")
    print("Gunakan kolom attrition_risk_score untuk prioritas intervensi HR.")


if __name__ == "__main__":
    main()
