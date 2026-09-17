# Project Speech To Text & Diarization Rapat (Faster-Whisper - Fast Mode)

Aplikasi Speech-to-Text cerdas yang dirancang khusus untuk **kebutuhan rapat/meeting panjang (> 2 Jam)**, dilengkapi dengan **Speaker Diarization**, **Noise Reduction**, **Greedy Search (beam_size=1)**, alokasi hemat daya **4 Core CPU**, dan **Progress Bar dinamis** tanpa membebani layar terminal.

---

## Fitur & Optimasi

1. **Alokasi 4 Core CPU (`--cpu-threads 4`)**:
   - Dibatasi hanya menggunakan 4 core prosesor secara default agar laptop tetap adem, hemat daya, dan Anda tetap bisa multitasking membuka aplikasi lain dengan lancar.
2. **Optimasi Waktu (beam_size=1)**:
   - Menggunakan mode *Greedy Search* (`beam_size=1`) yang memangkas waktu proses hingga **3x lebih cepat** tanpa penurunan akurasi yang berarti.
3. **Terminal Bersih & Ringan (Live Progress Bar)**:
   - Teks transkripsi **tidak lagi membanjiri layar terminal**, melainkan langsung ditulis (*streaming*) ke dalam file `.txt`.
   - Di terminal hanya tampil **1 baris Progress Bar interaktif** yang menampilkan: persentase (%), menit audio yang sedang diproses vs total durasi, kecepatan (x), dan estimasi sisa waktu.
4. **Speaker Diarization (Pengenal Pembicara)**:
   - Membedakan pembicara otomatis (`[Pembicara 1]`, `[Pembicara 2]`, dst.).
   - Menggunakan model PyAnnote Segmentation 3.0 + 3D-Speaker ONNX (100% lokal di CPU).
5. **Noise Reduction & VAD**:
   - Membersihkan desis mikrofon, AC, kipas angin, dan membuang jeda hening rapat.
6. **Notulensi Rapat Otomatis Per-Menu (Google Gemini AI)**:
   - Menghasilkan ringkasan notulensi eksekutif terstruktur rapi ke file `.md`.
   - Mengelompokkan poin bahasan, kendala, dan keputusan **per-menu aplikasi/modul sistem**, dilengkapi tabel **Action Items (Tugas, PIC, Target)**.
   - Menggunakan API Google Gemini 2.5/3.6 Flash yang cepat dan gratis tanpa kartu kredit.
7. **Manajemen File Rapi Otomatis (`Hasil/<Nama Rekaman>/`)**:
   - Semua file luaran (transkrip `.txt`, notulensi `.md`, audio `.wav`) otomatis dikelompokkan ke dalam folder `Hasil/[Nama Rekaman]/` sehingga folder utama tetap bersih dan rapi.

---

## Cara Menjalankan

1. **Masuk ke folder proyek & aktifkan venv:**
   ```powershell
   cd "C:\Spech To Text"
   .\venv\Scripts\activate
   ```

2. **Jalankan Transkripsi Rapat (Mode 4 Core Otomatis):**
   ```powershell
   python transcribe.py "D:\Meeting\rekaman_rapat.mp3"
   ```
   *Secara default otomatis menggunakan model `small`, `beam_size=1`, 4 core CPU, bahasa Indonesia (`id`), dan hasil langsung tersimpan rapi di `Hasil/rekaman_rapat/rekaman_rapat_transkripsi.txt`.*

3. **Transkripsi Sekaligus Buat Notulensi Rapat Per-Menu (Gemini AI):**
   ```powershell
   python transcribe.py "rapat.mp3" --summarize
   ```

4. **Meringkas File Transkripsi yang Sudah Ada (Tanpa Transkripsi Ulang):**
   ```powershell
   python transcribe.py "Hasil/rapat/rapat_transkripsi.txt"
   ```

5. **Opsi Tambahan untuk Meningkatkan Akurasi:**
   - **Kamus Istilah Rapat / Nama Orang (`--prompt`)**:
     ```powershell
     python transcribe.py "rapat.mp3" --prompt "Pak Dida, Pak Woko, Master Produk, Kasir Penjualan, SSD, SQL"
     ```
   - **Mode Pencarian Akurasi Tinggi (`--beam-size 5`)**:
     ```powershell
     python transcribe.py "rapat.mp3" --beam-size 5
     ```
   - **Tentukan jumlah peserta jika diketahui (misal 3 orang):**
     ```powershell
     python transcribe.py "rapat.mp3" --num-speakers 3
     ```
   - **Ingin tetap menampilkan teks di terminal sambil berjalan?**
     ```powershell
     python transcribe.py "rapat.mp3" --show-text
     ```
   - **Jika rekaman sudah sangat jernih (rekaman Zoom/Meet) dan ingin lebih cepat lagi:**
     ```powershell
     python transcribe.py "rapat.mp3" --no-noise-reduction
     ```

---

## Setup Google Gemini API (Gratis)
1. Kunjungi [Google AI Studio](https://aistudio.google.com/) dan login dengan akun Google Anda.
2. Klik **"Get API key"** -> **"Create API key"** dan salin API key Anda.
3. Buat file `.env` di folder proyek ini (sejajar dengan `transcribe.py`) dan isi:
   ```env
   GEMINI_API_KEY=AIzaSy...
   ```
   *(File `.env` otomatis diabaikan oleh git sehingga kunci rahasia Anda tetap aman).*

---

## Author & Kontribusi
Dibuat dan dikembangkan oleh [Ilyas Abd Rohim](https://github.com/IlyasAbdRohim). Silakan berkontribusi atau membuat issue jika menemukan kendala.
