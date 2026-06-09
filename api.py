import pandas as pd
import numpy as np
import os
from io import StringIO
from flask import Flask, jsonify
from flask_cors import CORS
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)


def process_data(file_path):
    print("\n📂 Membaca file Excel...")
    
    df_raw = pd.read_excel(file_path, header=0)
    df_raw.columns = df_raw.columns.astype(str).str.strip()
    
    if df_raw.iloc[0].notna().sum() > len(df_raw.columns) / 2:
        df_raw.columns = df_raw.iloc[0]
        df_raw = df_raw.iloc[1:]
        df_raw.reset_index(drop=True, inplace=True)
    
    print("✅ File berhasil dibaca.")
    
    df_desa = df_raw.iloc[:, 0:36]
    df_puskesmas = df_raw.iloc[:, 37:56]
    df_kecamatan = df_raw.iloc[:, 57:75]
    df_kabupaten = df_raw.iloc[:, 76:93]
    
    def clean_basic(df):
        df = df.dropna(how="all")
        df = df.dropna(axis=1, how="all")
        
        if len(df) > 0 and df.iloc[0, 0] == df.columns[0]:
            df = df.iloc[1:]
        return df
    
    df_desa = clean_basic(df_desa)
    df_puskesmas = clean_basic(df_puskesmas)
    df_kecamatan = clean_basic(df_kecamatan)
    df_kabupaten = clean_basic(df_kabupaten)
    
    def clean_region(df):
        for col in ["Kabupaten", "Kecamatan", "PUSKESMAS", "DESA/KELURAHAN"]:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .str.replace(",", "", regex=False)
                )
        return df
    
    df_desa = clean_region(df_desa)
    df_puskesmas = clean_region(df_puskesmas)
    df_kecamatan = clean_region(df_kecamatan)
    df_kabupaten = clean_region(df_kabupaten)
    
    def clean_time(df):
        if "Tahun" in df.columns:
            df["Tahun"] = pd.to_numeric(df["Tahun"], errors="coerce").astype("Int64")
        
        if "Bulan" in df.columns:
            df["Bulan_Bersih"] = df["Bulan"].astype(str).str.strip().str.lower()
            
            bulan_map = {
                "januari": 1,
                "februari": 2,
                "maret": 3,
                "april": 4,
                "mei": 5,
                "juni": 6,
                "juli": 7,
                "agustus": 8,
                "september": 9,
                "oktober": 10,
                "november": 11,
                "desember": 12
            }
            
            df["Angka_Bulan"] = (
                df["Bulan_Bersih"]
                .map(bulan_map)
                .astype("Int64")
            )
        
        return df
    
    df_desa = clean_time(df_desa)
    df_puskesmas = clean_time(df_puskesmas)
    df_kecamatan = clean_time(df_kecamatan)
    df_kabupaten = clean_time(df_kabupaten)
    
    def clean_wilayah(df, kolom_desa=None, kolom_puskesmas=None, khusus_desa=False):
        
        def bersihkan_umum(teks):
            return (
                teks.astype(str)
                .str.replace("KAB\\.", "", regex=True)
                .str.strip()
                .str.lower()
                .str.replace(r"\s+", "", regex=True)
                .str.capitalize()
            )
        
        def bersihkan_desa(teks):
            teks = teks.astype(str).str.lower().str.strip()
            teks_no_space = teks.str.replace(r"\s+", "", regex=True)
            
            pengecualian = {
                "olakalen": "Olak Alen"
            }
            
            teks_final = teks_no_space.replace(pengecualian)
            
            mask = ~teks_no_space.isin(pengecualian.keys())
            teks_final.loc[mask] = (
                teks_no_space.loc[mask]
                .str.capitalize()
            )
            
            return teks_final
        
        if "Kabupaten" in df.columns:
            df["Kabupaten"] = bersihkan_umum(df["Kabupaten"])
        
        if "Kecamatan" in df.columns:
            df["Kecamatan"] = bersihkan_umum(df["Kecamatan"])
        
        if kolom_puskesmas and kolom_puskesmas in df.columns:
            df[kolom_puskesmas] = bersihkan_umum(df[kolom_puskesmas])
        
        if kolom_desa and kolom_desa in df.columns:
            if khusus_desa:
                df[kolom_desa] = bersihkan_desa(df[kolom_desa])
            else:
                df[kolom_desa] = bersihkan_umum(df[kolom_desa])
        
        return df
    
    df_desa = clean_wilayah(
        df_desa,
        kolom_desa="DESA/KELURAHAN",
        kolom_puskesmas="PUSKESMAS",
        khusus_desa=True
    )
    
    df_puskesmas = clean_wilayah(
        df_puskesmas,
        kolom_puskesmas="PUSKESMAS"
    )
    
    df_kecamatan = clean_wilayah(df_kecamatan)
    df_kabupaten = clean_wilayah(df_kabupaten)
    
    def drop_unnamed(df):
        return df.loc[:, ~df.columns.str.contains("^Unnamed", na=False)]
    
    df_desa = drop_unnamed(df_desa)
    df_puskesmas = drop_unnamed(df_puskesmas)
    df_kecamatan = drop_unnamed(df_kecamatan)
    df_kabupaten = drop_unnamed(df_kabupaten)
    
    for df in [df_puskesmas, df_kecamatan, df_kabupaten]:
        if "Undeweight" in df.columns:
            df.rename(
                columns={"Undeweight": "Underweight"},
                inplace=True
            )
    
    def rename_pair_columns(df):
    
        df = df.copy()
        cols = list(df.columns)
        new_cols = []
        
        i = 0
        
        while i < len(cols):
            col = cols[i]
            
            if col == "%" and i > 0:
                prev_col = new_cols[-1]
                new_cols.append("%" + prev_col)
            else:
                new_cols.append(col)
            
            i += 1
        
        df.columns = new_cols
        return df
    
    df_puskesmas = rename_pair_columns(df_puskesmas)
    df_kecamatan = rename_pair_columns(df_kecamatan)
    df_kabupaten = rename_pair_columns(df_kabupaten)
    
    if "%underweight" in df_desa.columns:
        df_desa.rename(
            columns={"%underweight": "%Underweight"},
            inplace=True
        )
    
    def make_columns_unique(df):
        
        df = df.copy()
        cols = pd.Series(df.columns)
        
        for dup in cols[cols.duplicated()].unique():
            indices = cols[cols == dup].index
            
            for i, idx in enumerate(indices):
                if i == 0:
                    cols.iloc[idx] = dup
                else:
                    cols.iloc[idx] = f"{dup}_{i}"
        
        df.columns = cols
        return df
    
    df_desa = make_columns_unique(df_desa)
    df_puskesmas = make_columns_unique(df_puskesmas)
    df_kecamatan = make_columns_unique(df_kecamatan)
    df_kabupaten = make_columns_unique(df_kabupaten)
    
    # ========================
    # KODE WILAYAH DARI CSV (TAMBAHAN)
    # ========================
    print("📂 Membaca file CSV kode wilayah...")
    
    # Fungsi bersihkan umum yang sama (agar konsisten)
    def bersihkan_umum_csv(teks):
        return (
            teks.astype(str)
            .str.replace("KAB\\.", "", regex=True)
            .str.strip()
            .str.lower()
            .str.replace(r"\s+", "", regex=True)
            .str.capitalize()
        )
    
    # Baca dan bersihkan CSV desa
    df_kode_desa = pd.read_csv("Kode wilayah desa_kelurahan.csv", dtype={"kode_kemendagri": str})
    print(f"  - Desa CSV: {len(df_kode_desa)} baris")
    if "nama_wilayah" in df_kode_desa.columns:
        df_kode_desa["nama_wilayah"] = bersihkan_umum_csv(df_kode_desa["nama_wilayah"])
    else:
        print("⚠️ CSV desa tidak memiliki kolom 'nama_wilayah'")
    
    # Baca dan bersihkan CSV kecamatan
    df_kode_kecamatan = pd.read_csv("kode wilayah kecamatan.csv", dtype={"kode_kemendagri": str})
    print(f"  - Kecamatan CSV: {len(df_kode_kecamatan)} baris")
    if "nama_wilayah" in df_kode_kecamatan.columns:
        df_kode_kecamatan["nama_wilayah"] = bersihkan_umum_csv(df_kode_kecamatan["nama_wilayah"])
    else:
        print("⚠️ CSV kecamatan tidak memiliki kolom 'nama_wilayah'")
    
    # Mapping kecamatan dari CSV
    kode_kecamatan_dict = df_kode_kecamatan.set_index("nama_wilayah")["kode_kemendagri"].to_dict()
    
    def get_kode_kecamatan(nama_kec):
        if pd.isna(nama_kec) or nama_kec == "":
            return None
        nama_clean = str(nama_kec).strip().lower()
        # Exact match
        for k, v in kode_kecamatan_dict.items():
            if k.lower() == nama_clean:
                return v
        # Fallback substring
        for k, v in kode_kecamatan_dict.items():
            if nama_clean in k.lower() or k.lower() in nama_clean:
                print(f"  - Fallback kec: '{nama_kec}' -> '{k}' ({v})")
                return v
        print(f"⚠️ Kecamatan tidak ditemukan: '{nama_kec}'")
        return None
    
    # Tambahkan kolom kode_kecamatan sementara untuk desa
    df_desa["kode_kecamatan"] = df_desa["Kecamatan"].apply(get_kode_kecamatan)
    print(f"Null kode_kecamatan pada desa: {df_desa['kode_kecamatan'].isna().sum()}/{len(df_desa)}")
    
    # Untuk desa: ekstrak kode kecamatan dari 6 digit pertama kode desa
    if "kode_kemendagri" in df_kode_desa.columns:
        df_kode_desa["kode_kecamatan"] = df_kode_desa["kode_kemendagri"].str[:6]
        # Merge desa dengan kode desa
        df_desa = df_desa.merge(
            df_kode_desa[["nama_wilayah", "kode_kemendagri", "kode_kecamatan"]],
            left_on=["DESA/KELURAHAN", "kode_kecamatan"],
            right_on=["nama_wilayah", "kode_kecamatan"],
            how="left"
        )
        df_desa.rename(columns={"kode_kemendagri": "kode_wilayah"}, inplace=True)
        df_desa.drop(columns=["nama_wilayah"], inplace=True, errors="ignore")
    else:
        print("⚠️ CSV desa tidak memiliki kolom 'kode_kemendagri', kode wilayah desa tidak dapat diisi")
        df_desa["kode_wilayah"] = ""
    
    # Untuk kecamatan
    df_kecamatan["kode_wilayah"] = df_kecamatan["Kecamatan"].apply(get_kode_kecamatan)
    
    # Untuk kabupaten: ambil kode dari kecamatan yang mengandung "kab"
    kab_list = df_kode_kecamatan[df_kode_kecamatan["nama_wilayah"].str.contains("kab", na=False)]
    if not kab_list.empty:
        kode_kab = kab_list.iloc[0]["kode_kemendagri"][:4]
        df_kabupaten["kode_wilayah"] = kode_kab
    else:
        df_kabupaten["kode_wilayah"] = ""
    
    # Untuk puskesmas tidak ada kode wilayah
    df_puskesmas["kode_wilayah"] = ""
    
    # Tampilkan contoh mapping desa
    contoh = df_desa[df_desa["kode_wilayah"].notna()][["DESA/KELURAHAN", "Kecamatan", "kode_wilayah"]].head(3)
    print("\n✅ Contoh mapping desa berhasil:")
    print(contoh.to_string())
    # =========================================================
    
    # Lanjut ke penggabungan
    df_desa["Level"] = "Desa"
    df_puskesmas["Level"] = "Puskesmas"
    df_kecamatan["Level"] = "Kecamatan"
    df_kabupaten["Level"] = "Kabupaten"
    
    df_final = pd.concat(
        [
            df_desa,
            df_puskesmas,
            df_kecamatan,
            df_kabupaten
        ],
        ignore_index=True
    )
    
    df_final["Periode"] = (
        df_final["Tahun"].astype(str)
        + "-"
        + df_final["Angka_Bulan"].astype(str).str.zfill(2)
    )
    
    df_final["Wilayah"] = (
        df_final["DESA/KELURAHAN"]
        .fillna(df_final["PUSKESMAS"])
        .fillna(df_final["Kecamatan"])
        .fillna(df_final["Kabupaten"])
    )
    
    indikator = [
        "Stunting",
        "%Stunting",
        "Wasting",
        "%Wasting",
        "Underweight",
        "%Underweight",
        "Overweight",
        "%Overweight"
    ]
    
    kolom_detail_desa = [
        "Kecamatan",
        "PUSKESMAS",
        "DESA/KELURAHAN",
        "D/S",
        "Sangat Kurang",
        "Kurang",
        "Berat Badan Normal",
        "Risiko Lebih",
        "Sangat Pendek",
        "Pendek",
        "Normal",
        "Tinggi",
        "Gizi Buruk",
        "Gizi Kurang",
        "Normal_1",
        "Risiko Gizi Lebih",
        "Gizi Lebih",
        "Obesitas"
    ]
    
    for col in indikator + kolom_detail_desa:
        if col not in df_final.columns:
            df_final[col] = ""
    
    # Masukkan kolom kode_wilayah ke df_dashboard
    df_dashboard = df_final[
        [
            "Periode",
            "Tahun",
            "Bulan",
            "Level",
            "Wilayah"
        ]
        + indikator
        + kolom_detail_desa
        + ["kode_wilayah"]   # <-- tambahan kolom kode wilayah
    ].copy()
    
    cols_persen = [
        "%Stunting",
        "%Wasting",
        "%Underweight",
        "%Overweight"
    ]
    
    for col in cols_persen:
        if col in df_dashboard.columns:
            
            df_dashboard[col] = (
                df_dashboard[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .str.replace("%", "", regex=False)
            )
            
            df_dashboard[col] = (
                pd.to_numeric(
                    df_dashboard[col],
                    errors="coerce"
                )
                .fillna(0)
            )
    
    mask_kab = df_dashboard["Level"] == "Kabupaten"
    
    for col in kolom_detail_desa:
        if col in df_dashboard.columns:
            df_dashboard.loc[mask_kab, col] = ""
    
    # =========================================================
    # K-MEANS CLUSTERING
    # =========================================================
    
    fitur_cluster = [
        "%Stunting",
        "%Wasting",
        "%Underweight",
        "%Overweight"
    ]
    
    df_cluster = df_dashboard.copy()
    
    X = df_cluster[fitur_cluster].fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )
    
    df_cluster["Cluster"] = kmeans.fit_predict(X_scaled)
    
    nama_cluster = {
        0: "Kondisi Gizi Relatif Baik",
        1: "Perlu Pemantauan Gizi",
        2: "Prioritas Pendampingan Gizi",
        3: "Fokus Intervensi Kesehatan"
    }
    
    df_cluster["Nama_Cluster"] = (
        df_cluster["Cluster"]
        .map(nama_cluster)
    )
    
    df_dashboard["Cluster"] = df_cluster["Cluster"]
    df_dashboard["Nama_Cluster"] = df_cluster["Nama_Cluster"]
    
    # =========================================================
    
    df_dashboard = df_dashboard.fillna("")
    
    print(f"✅ Data diproses: {len(df_dashboard)} baris")
    print("✅ K-Means Clustering berhasil ditambahkan")
    
    return df_dashboard


@app.route("/api/data")
def api_data():
    
    try:
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        file_path = os.path.join(
            base_dir,
            "Data Berjalan RAW.xlsx"
        )
        
        if not os.path.exists(file_path):
            return jsonify({
                "error": f"File tidak ditemukan: {file_path}"
            }), 404
        
        df = process_data(file_path)
        
        records = df.to_dict(orient="records")
        
        for rec in records:
            for k, v in rec.items():
                if pd.isna(v):
                    rec[k] = None
        
        return jsonify(records)
    
    except Exception as e:
        
        print("❌ ERROR:", str(e))
        
        import traceback
        traceback.print_exc()
        
        return jsonify({
            "error": f"Gagal proses data: {str(e)}"
        }), 500


if __name__ == "__main__":
    
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )
