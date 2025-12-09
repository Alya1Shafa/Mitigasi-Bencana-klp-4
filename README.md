📊 Analisis Sentimen Bencana Alam Indonesia (X/Twitter Data)
Proyek ini bertujuan untuk mengembangkan model klasifikasi sentimen yang mampu memproses dan menganalisis reaksi publik di platform X (Twitter) terhadap tiga jenis bencana alam yang sering terjadi di Indonesia: Banjir, Tanah Longsor, dan Gempa Bumi.

Data diambil secara real-time menggunakan X API V2.

⚙️ Struktur Proyek
Repositori ini berisi kode inti untuk pengambilan data (data scraping) dan pemodelan Machine Learning:

get_data_scripts/: Berisi script Python untuk pengambilan data (API).

get_flood_data.py: Skrip untuk mengambil data Banjir.

get_landslide_data.py: Skrip untuk mengambil data Tanah Longsor.

get_earthquake_data.py: Skrip untuk mengambil data Gempa Bumi.

data/: Folder tempat menyimpan dataset CSV yang sudah digabungkan.

banjir_data_realtime.csv

longsor_data_realtime.csv

gempa_data_realtime.csv

sentiment_analysis.py: Skrip utama untuk melatih model, menggabungkan ketiga dataset, menerapkan SMOTE, dan mengevaluasi kinerja model.

🎯 Metodologi Analisis
Kami menggunakan pendekatan Machine Learning untuk mengatasi tantangan ketidakseimbangan kelas (banyak tweet Netral, sedikit tweet Positif).

1. Penggabungan Data (Data Concatenation)
Ketiga dataset (Banjir, Longsor, Gempa) digabungkan menjadi satu dataset utama untuk menciptakan model sentimen yang umum untuk semua jenis bencana.

2. Pelabelan Sentimen (Simulasi)
Data tweet dilabeli secara otomatis menjadi tiga kelas (Negative, Positive, Neutral) berdasarkan kata kunci Bahasa Indonesia yang ada dalam teks.

3. Teknik Pemodelan Lanjutan
Vectorization: Menggunakan TF-IDF (Term Frequency-Inverse Document Frequency) dengan stop words Bahasa Indonesia untuk mengubah teks menjadi fitur numerik.

Penyeimbangan Kelas: Menerapkan teknik SMOTE (Synthetic Minority Over-sampling Technique) pada data pelatihan untuk menyeimbangkan kelas Minoritas (Negative dan Positive), yang bertujuan meningkatkan kemampuan model mengenali sentimen yang jarang muncul.

Perbandingan Model: Menguji dan membandingkan kinerja:

Complement Naive Bayes (CNB)

Logistic Regression (LR)

🚀 Cara Menggunakan Proyek Ini
Prasyarat
Python: Versi 3.10 ke atas (direkomendasikan menggunakan Virtual Environment (myenv)).

Akses X API: Bearer Token yang valid (harus terikat pada Proyek).

Instalasi Library
Pastikan lingkungan virtual Anda aktif, lalu instal semua library yang diperlukan:

Bash

(myenv) pip install tweepy pandas scikit-learn numpy imbalanced-learn tabulate
Alur Kerja
Pengambilan Data: Ganti BEARER_TOKEN di script get_data_scripts/*.py dengan token Anda, lalu jalankan satu per satu untuk mendapatkan ketiga file CSV di folder data/.

Analisis Sentimen: Jalankan script utama:

Bash

(myenv) python sentiment_analysis.py
Hasil: Script akan menampilkan laporan klasifikasi perbandingan model CNB dan LR, menunjukkan model mana yang memiliki kinerja terbaik setelah penyeimbangan SMOTE.

📧 Kontak
Jika Anda memiliki pertanyaan atau ingin berkolaborasi, silakan hubungi saya atau buka Issue di repositori ini.

Muhammad [Nama Anda]
