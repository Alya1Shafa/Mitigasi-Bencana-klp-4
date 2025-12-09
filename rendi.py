import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob

# ============================
# Fungsi analisis sentimen
# ============================
def analisis_sentimen(teks):
    if not isinstance(teks, str) or teks.strip() == "":
        return "Netral"

    score = TextBlob(teks).sentiment.polarity

    if score > 0.05:
        return "Positif"
    elif score < -0.05:
        return "Negatif"
    else:
        return "Netral"


# ============================
# Daftar file CSV
# ============================
file_bencana = {
    "Banjir": "banjir_data_realtime.csv",
    "Gempa Bumi": "gempa_data_realtime.csv",
    "Tanah Longsor": "longsor_data_realtime.csv"
}

data_gabungan = []

# ============================
# Membaca & memproses data
# ============================
for kategori, file in file_bencana.items():
    print(f"\n=== Memproses data: {kategori} ===")

    try:
        df = pd.read_csv(file)

        if "Text" not in df.columns:
            print(f"❌ Kolom 'Text' tidak ditemukan dalam {file}")
            continue

        df["Sentimen"] = df["Text"].apply(analisis_sentimen)
        df["Kategori"] = kategori

        data_gabungan.append(df)

    except Exception as e:
        print(f"Error saat membaca {file}: {e}")

if len(data_gabungan) == 0:
    print("\n❌ Tidak ada data yang bisa dianalisis!")
    exit()

data_final = pd.concat(data_gabungan, ignore_index=True)

# convert date
data_final["Date"] = pd.to_datetime(data_final["Date"], errors="coerce")

# Simpan hasil
data_final.to_csv("hasil_sentimen.csv", index=False)
print("\n✔ Hasil analisis disimpan sebagai hasil_sentimen.csv\n")


# ============================
# Persiapan Statistik
# ============================
stat_sentimen = (
    data_final.groupby(["Kategori", "Sentimen"])
    .size()
    .unstack(fill_value=0)
)

trend = (
    data_final.groupby([data_final["Date"].dt.date, "Kategori"])
    .size()
    .unstack(fill_value=0)
)

# ============================
# SEMUA GRAFIK DALAM SATU WINDOW
# ============================

fig, axs = plt.subplots(3, 2, figsize=(15, 15))
fig.suptitle("Analisis Sentimen & Statistik Bencana dari Data Twitter", fontsize=16)

# === Pie Chart Banjir ===
df_banjir = data_final[data_final["Kategori"] == "Banjir"]
df_banjir["Sentimen"].value_counts().plot(
    kind="pie", autopct="%1.1f%%", ax=axs[0, 0], legend=False
)
axs[0, 0].set_title("Sentimen Banjir")
axs[0, 0].set_ylabel("")

# === Pie Chart Gempa Bumi ===
df_gempa = data_final[data_final["Kategori"] == "Gempa Bumi"]
df_gempa["Sentimen"].value_counts().plot(
    kind="pie", autopct="%1.1f%%", ax=axs[0, 1], legend=False
)
axs[0, 1].set_title("Sentimen Gempa Bumi")
axs[0, 1].set_ylabel("")

# === Pie Chart Tanah Longsor ===
df_longsor = data_final[data_final["Kategori"] == "Tanah Longsor"]
df_longsor["Sentimen"].value_counts().plot(
    kind="pie", autopct="%1.1f%%", ax=axs[1, 0], legend=False
)
axs[1, 0].set_title("Sentimen Tanah Longsor")
axs[1, 0].set_ylabel("")

# === Grafik Batang Perbandingan Sentimen ===
stat_sentimen.plot(kind="bar", ax=axs[1, 1])
axs[1, 1].set_title("Perbandingan Sentimen Antar Bencana")
axs[1, 1].set_xlabel("Kategori Bencana")
axs[1, 1].set_ylabel("Jumlah Tweet")
axs[1, 1].legend(title="Sentimen")

# === Tren Tweet per Hari ===
trend.plot(ax=axs[2, 0])
axs[2, 0].set_title("Tren Jumlah Tweet per Hari")
axs[2, 0].set_xlabel("Tanggal")
axs[2, 0].set_ylabel("Jumlah Tweet")
axs[2, 0].tick_params(axis='x', rotation=45)

# === Kosongkan panel terakhir (atau bisa isi wordcloud bila kamu mau) ===
axs[2, 1].axis("off")
axs[2, 1].set_title("Kosong (bisa isi WordCloud)")

plt.tight_layout()
plt.show()
