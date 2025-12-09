import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import os

# ======================================================
# Fungsi untuk menentukan sentimen (positif/negatif/netral)
# ======================================================
def get_sentiment(text):
    analysis = TextBlob(str(text))
    if analysis.sentiment.polarity > 0:
        return "Positif"
    elif analysis.sentiment.polarity < 0:
        return "Negatif"
    else:
        return "Netral"

# ======================================================
# Daftar file CSV (ganti nama file sesuai punya kamu)
# ======================================================
file_paths = {
    "Banjir": "banjir_data_realtime.csv",
    "Gempa Bumi": "gempa_data_realtime.csv",
    "Tanah Longsor": "longsor.csv"
}

# Menyimpan hasil semua kategori
summary_sentimen = {}

# ======================================================
# Proses tiap dataset bencana
# ======================================================
for category, filepath in file_paths.items():

    if not os.path.exists(filepath):
        print(f"File tidak ditemukan: {filepath}")
        continue

    print(f"\n=== Memproses: {category} ===")

    df = pd.read_csv(filepath)

    # Pastikan kolom tweet bernama 'tweet'
    if "tweet" not in df.columns:
        print(f"Kolom 'tweet' tidak ditemukan dalam file {filepath}")
        continue
    
    # Hitung sentimen
    df["sentimen"] = df["tweet"].apply(get_sentiment)

    # Hitung distribusi sentimen
    counts = df["sentimen"].value_counts()
    summary_sentimen[category] = counts

# ======================================================
# VISUALISASI 1 — Pie Chart per Bencana
# ======================================================
for category, counts in summary_sentimen.items():

    plt.figure(figsize=(6, 6))
    plt.title(f"Distribusi Sentimen untuk {category}")
    plt.pie(counts, labels=counts.index, autopct="%1.1f%%")
    plt.show()

# ======================================================
# VISUALISASI 2 — Bar Chart Perbandingan Antar Bencana
# ======================================================

# Membuat DataFrame gabungan
summary_df = pd.DataFrame(summary_sentimen).fillna(0)

plt.figure(figsize=(10, 6))
summary_df.T.plot(kind="bar")
plt.title("Perbandingan Sentimen Antar Bencana")
plt.xlabel("Jenis Bencana")
plt.ylabel("Jumlah Tweet")
plt.legend(title="Kategori Sentimen")
plt.xticks(rotation=0)
plt.show()
