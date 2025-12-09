import tweepy
import pandas as pd
import time
import sys

# =========================================================
# ⚠️ LANGKAH PENTING: GANTI DENGAN BEARER TOKEN BARU ⚠️
# Token ini harus didapatkan dari App yang terikat pada PROJECT di X Developer Portal Anda.
BEARER_TOKEN = "PASTE_TOKEN_BARU_DI_SINI" 
# =========================================================

# --- 1. Inisialisasi Klien Tweepy ---
def initialize_client(token):
    try:
        client = tweepy.Client(token)
        # Coba permintaan sederhana untuk verifikasi token
        client.get_users(usernames=['TwitterDev']) 
        return client
    except Exception as e:
        print("\nERROR: Gagal menginisialisasi klien.")
        print("Pastikan Bearer Token Anda sudah benar dan aktif.")
        sys.exit(1)

# --- 2. Fungsi Utama Pengambilan Data ---
def fetch_earthquake_data(client, query, max_tweets_total):
    tweets_data = []
    print(f"Mulai mengambil hingga {max_tweets_total} tweet tentang Gempa Bumi...")

    try:
        # Menggunakan search_recent_tweets (kompatibel dengan akses dasar)
        # Mengambil data dari 7 hari terakhir.
        response = client.search_recent_tweets(
            query=query,
            max_results=100, # Maksimal 100 per request pada akses dasar
            tweet_fields=["created_at", "public_metrics"]
        )

        # Mengumpulkan data
        if response.data:
            for tweet in response.data:
                # Batasi hingga max_tweets_total jika lebih dari satu halaman (walaupun di sini hanya satu req)
                if len(tweets_data) >= max_tweets_total:
                    break 
                
                tweets_data.append([
                    tweet.text,
                    tweet.created_at,
                    tweet.public_metrics.get('like_count', 0),
                    tweet.public_metrics.get('retweet_count', 0)
                ])
            print(f"Berhasil mengambil {len(tweets_data)} tweet terbaru.")
        else:
            print("Peringatan: Tidak ada tweet yang ditemukan dengan query ini (mungkin karena filter waktu atau kueri terlalu spesifik).")

    except tweepy.errors.TweepyException as e:
        print(f"\nERROR API Tweepy: Gagal saat mencari tweet.")
        print(f"Pesan: {e}")
        # Jika masih 403, Token Anda BUKAN dari Proyek.
        sys.exit(1)
        
    return tweets_data

# --- 3. Fungsi Penyimpanan dan Eksekusi ---
if __name__ == "__main__":
    
    # 2. Parameter Pencarian
    QUERY = 'gempa OR "gempa bumi" OR guncangan OR kerusakan lang:id -is:retweet' 
    MAX_TWEETS_TO_FETCH = 100 # Sesuaikan karena max_results saat ini 100.

    # 1. Inisialisasi Klien
    client = initialize_client(BEARER_TOKEN)
    
    # 2. Ambil Data
    tweets_data = fetch_earthquake_data(client, QUERY, MAX_TWEETS_TO_FETCH)
    
    # 3. Simpan ke CSV
    df = pd.DataFrame(tweets_data, columns=['Text', 'Date', 'Likes', 'Retweets'])
    output_file = 'gempa_data_realtime.csv' 
    df.to_csv(output_file, index=False)

    print("\n" + "="*50)
    print(f"STATUS: Berhasil! Data tersimpan di: {output_file}")
    print("==================================================")