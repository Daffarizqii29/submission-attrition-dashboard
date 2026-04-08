# Proyek Akhir: Menyelesaikan Permasalahan Human Resources

## Business Understanding

Jaya Jaya Maju merupakan perusahaan multinasional yang telah berdiri sejak tahun 2000 dan memiliki lebih dari 1.000 karyawan yang tersebar di berbagai wilayah. Seiring pertumbuhan perusahaan, tim HR menghadapi tantangan dalam menjaga stabilitas tenaga kerja karena tingkat attrition perusahaan telah melampaui 10%.

Tingginya attrition dapat berdampak pada biaya rekrutmen, produktivitas tim, transfer pengetahuan, dan kesinambungan operasional. Oleh karena itu, perusahaan membutuhkan analisis data untuk memahami faktor-faktor yang berkaitan dengan attrition, dashboard bisnis untuk memantau kondisi karyawan, serta model prediksi untuk membantu prioritas intervensi retensi.

### Permasalahan Bisnis

1. Perusahaan perlu memahami faktor-faktor yang berkaitan dengan tingginya attrition karyawan.
2. Tim HR memerlukan dashboard bisnis untuk memantau indikator dan segmen karyawan yang perlu diperhatikan.
3. Perusahaan membutuhkan model prediksi untuk membantu identifikasi dini karyawan yang memiliki risiko attrition lebih tinggi.

---

## Cakupan Proyek

- Melakukan data understanding dan data cleaning pada data HR.
- Menganalisis faktor yang paling berkaitan dengan attrition pada data berlabel.
- Membuat business dashboard yang menampilkan KPI serta faktor utama attrition.
- Membangun model machine learning klasifikasi attrition.
- Menyusun prototype prediksi sederhana berbasis file CSV.
- Menuliskan kesimpulan dan action items berbasis data.

---

## Persiapan

### Sumber data
Dataset yang digunakan adalah file dataset yang disediakan pada halaman submission Dicoding dan disimpan pada repository ini sebagai:

- [employee_data.csv](data/employee_data.csv)

---

### Setup environment

Proyek ini diuji menggunakan **Python 3.13.5**.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Untuk Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

---

### Menjalankan notebook

```bash
jupyter notebook notebook.ipynb
```

---

### Menjalankan dashboard Streamlit secara lokal

```bash
streamlit run dashboard/streamlit_app.py
```

Catatan:
- Dashboard akan tetap berjalan walaupun file model tersimpan tidak kompatibel dengan environment saat ini, karena aplikasi akan melatih ulang model secara otomatis dari dataset.

---

### Menjalankan prototype prediksi

```bash
python prediction.py --input examples/input_3_karyawan.csv --output examples/output_3_karyawan.csv
```

---

## Business Dashboard

Dashboard dapat dijalankan dari file berikut:

- `dashboard/streamlit_app.py`

Dashboard menampilkan:

- attrition rate keseluruhan
- ringkasan jumlah data berlabel dan label kosong
- overtime
- business travel
- job role berisiko tinggi
- kelompok usia
- kelompok pendapatan
- masa kerja
- environment satisfaction
- segmen prioritas intervensi

Screenshot dashboard:
`daffa1212_dicoding-dashboard.png`

---

## Hasil Analisis

Temuan utama dari analisis data berlabel adalah sebagai berikut:

1. Attrition rate keseluruhan mencapai **16.92%**.
2. Karyawan dengan **OverTime = Yes** memiliki attrition rate lebih tinggi dibandingkan karyawan tanpa lembur.
3. Karyawan dengan frekuensi perjalanan bisnis tertentu menunjukkan risiko attrition yang lebih tinggi.
4. Kelompok usia yang lebih muda cenderung memiliki attrition rate lebih tinggi.
5. Kelompok pendapatan rendah dan masa kerja awal merupakan segmen yang perlu dipantau lebih dekat.
6. Job role tertentu, seperti **Sales Representative** dan **Laboratory Technician**, muncul sebagai kelompok dengan attrition lebih tinggi.
7. Tingkat **EnvironmentSatisfaction** yang rendah berkaitan dengan attrition yang lebih tinggi.

---

## Modeling

Model yang digunakan adalah **Extra Trees Classifier** dengan preprocessing:

- imputasi median untuk fitur numerik
- imputasi modus untuk fitur kategorikal
- standardisasi fitur numerik
- one-hot encoding untuk fitur kategorikal

### Performa Model

- Accuracy: **0.854**
- Precision: **0.576**
- Recall: **0.528**
- F1-score: **0.551**
- ROC-AUC: **0.804**

### Driver Utama Model

1. OverTime
2. JobRole
3. EnvironmentSatisfaction
4. MaritalStatus
5. JobSatisfaction
6. EducationField

---

## Conclusion

Attrition di Jaya Jaya Maju dipengaruhi oleh kombinasi faktor beban kerja, kondisi kerja, pendapatan, dan masa kerja. Risiko attrition tidak tersebar merata pada seluruh karyawan, tetapi lebih terkonsentrasi pada segmen tertentu seperti karyawan yang sering lembur, berada pada masa kerja awal, memiliki pendapatan lebih rendah, atau memiliki tingkat kepuasan lingkungan kerja yang rendah.

Dashboard membantu tim HR memantau faktor risiko tersebut secara berkala, sedangkan model prediksi dapat digunakan sebagai alat bantu untuk menentukan prioritas intervensi retensi.

---

## Rekomendasi Action Items

1. Fokus pada karyawan dengan overtime tinggi dan kompensasi yang relatif rendah.
2. Perkuat program onboarding dan retensi pada 2 tahun pertama masa kerja.
3. Lakukan evaluasi berkala pada unit dengan environment satisfaction rendah.
4. Gunakan model prediksi sebagai sistem peringatan dini untuk membantu prioritas tindak lanjut HR.
5. Review jalur karier dan strategi kompensasi pada job role dengan attrition tinggi.

---

## Panduan Penggunaan Prototype

### Format input
Gunakan file contoh:
- `examples/input_1_karyawan.csv`
- `examples/input_3_karyawan.csv`

### Prediksi

```bash
python prediction.py --input examples/input_3_karyawan.csv --output hasil.csv
```

### Output

File output akan menambahkan dua kolom berikut:
- `attrition_risk_score`
- `predicted_attrition`

Semakin tinggi `attrition_risk_score`, semakin tinggi prioritas intervensi HR.
