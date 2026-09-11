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
   *Secara default sudah otomatis menggunakan model `small`, `beam_size=1`, 4 core CPU, bahasa Indonesia (`id`), dan hasil langsung tersimpan ke `<nama_file>_transkripsi.txt`.*

3. **Opsi Tambahan:**
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
