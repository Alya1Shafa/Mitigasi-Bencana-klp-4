import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1) File input
FILES = {
    "banjir": "banjir_data_realtime.csv",
    "longsor": "longsor_data_realtime.csv",
    "gempa": "gempa_data_realtime.csv",
}

SENTIMENTS = ["Negative", "Neutral", "Positive"]
REQUIRED_COLS = {"Text", "Date", "Likes", "Retweets"}  # dipakai untuk validasi saja

# 2) Rule-based sentiment (SAMA seperti sebelumnya)
neg_kw = ['rusak','korban','parah','terjebak','butuh','bahaya','bantu','takut']
pos_kw = ['selamat','syukur','aman','pulih','doa','terima kasih']

# 3) Load + label sentimen
dfs = []
for name, path in FILES.items():
    df = pd.read_csv(path)

    # validasi kolom (biar aman)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"[{name}] Kolom wajib tidak ada: {missing}")

    # buang text kosong
    df["Text"] = df["Text"].astype(str).replace(r"^\s*$", np.nan, regex=True)
    df = df.dropna(subset=["Text"]).copy()

    # label kategori + sentimen
    df["Disaster_Type"] = name

    lower_text = df["Text"].astype(str).str.lower()
    is_neg = lower_text.apply(lambda t: any(k in t for k in neg_kw))
    is_pos = lower_text.apply(lambda t: any(k in t for k in pos_kw))

    df["Sentiment"] = np.where(is_neg, "Negative", np.where(is_pos, "Positive", "Neutral"))
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)

# 4) Tabel ringkas: counts & props (SAMA seperti sebelumnya)
counts = pd.crosstab(combined["Disaster_Type"], combined["Sentiment"]).reindex(columns=SENTIMENTS, fill_value=0)
props = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

order = list(FILES.keys())  # urutan kategori untuk plot

# 5) Kesimpulan di terminal (SAMA konsepnya)
overall = combined["Sentiment"].value_counts().reindex(SENTIMENTS, fill_value=0)
dom = overall.idxmax()
pct = (overall[dom] / max(overall.sum(), 1)) * 100

print("\n" + "="*80)
print("KESIMPULAN RINGKAS ANALISIS SENTIMEN (4 PANEL • SIMPEL)")
print("="*80)
print(f"1) Sentimen dominan keseluruhan: {dom} ({pct:.1f}%)")
print("\n2) Sentimen dominan per kategori bencana (berdasarkan proporsi):")
for ds in props.index:
    dom_ds = props.loc[ds].idxmax()
    val = props.loc[ds, dom_ds] * 100
    print(f"   - {ds}: {dom_ds} ({val:.1f}%)")
print("\nCatatan: Sentimen masih rule-based (keyword), sifatnya indikatif.")
print("="*80)

# 6) Visualisasi 4 panel (SAMA konsepnya)
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 2)

ax_pie   = fig.add_subplot(gs[0, 0])
ax_heat  = fig.add_subplot(gs[0, 1])
ax_stack = fig.add_subplot(gs[1, 0])
ax_over  = fig.add_subplot(gs[1, 1])

# (1) Pie overall
ax_pie.pie(overall.values, labels=overall.index.tolist(), autopct="%1.1f%%")
ax_pie.set_title("Komposisi Sentimen (Gabungan 3 Dataset)")

# (2) Heatmap proporsi per kategori
heat = props.reindex(index=order)
im = ax_heat.imshow(heat.values, aspect="auto", vmin=0, vmax=1)
ax_heat.set_title("Heatmap Proporsi Sentimen per Kategori")
ax_heat.set_yticks(range(len(heat.index)))
ax_heat.set_yticklabels(heat.index.tolist())
ax_heat.set_xticks(range(len(heat.columns)))
ax_heat.set_xticklabels(heat.columns.tolist())

for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        ax_heat.text(j, i, f"{heat.iloc[i, j]*100:.0f}%", ha="center", va="center")

fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

# (3) Stacked bar jumlah sentimen per kategori
c = counts.reindex(index=order)
bottom = np.zeros(len(c.index))
x = np.arange(len(c.index))
for s in SENTIMENTS:
    vals = c[s].values
    ax_stack.bar(x, vals, bottom=bottom, label=s)
    bottom += vals

ax_stack.set_title("Distribusi Sentimen per Kategori (Jumlah)")
ax_stack.set_xlabel("Kategori Bencana")
ax_stack.set_ylabel("Jumlah Tweet")
ax_stack.set_xticks(x)
ax_stack.set_xticklabels(c.index.tolist())
ax_stack.legend()

# (4) Bar overall (jumlah)
ax_over.bar(overall.index.tolist(), overall.values)
ax_over.set_title("Sentimen Keseluruhan (Jumlah)")
ax_over.set_xlabel("Sentimen")
ax_over.set_ylabel("Jumlah Tweet")

fig.suptitle("Dashboard Analisis Sentimen Pasca Bencana • Tutup window untuk selesai", fontsize=14)
fig.tight_layout(rect=[0, 0.02, 1, 0.95])

plt.show(block=True)
