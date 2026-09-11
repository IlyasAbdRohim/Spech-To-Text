import os
import sys
import time
import argparse

# Hilangkan peringatan symlink Windows pada Hugging Face Hub cache
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
from huggingface_hub import hf_hub_download
import noisereduce as nr
import soundfile as sf
import sherpa_onnx
import numpy as np

def format_timestamp(seconds: float) -> str:
    """Mengubah detik menjadi format HH:MM:SS"""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"

def render_progress_bar(
    current_val: float,
    total_val: float,
    start_time: float,
    prefix: str = "Progres",
    is_time: bool = True
):
    """Menampilkan indikator progres 1 baris interaktif di terminal"""
    if total_val <= 0:
        return
    pct = min(100.0, (current_val / total_val) * 100.0)
    bar_len = 25
    filled = int(bar_len * pct / 100)
    bar = "=" * filled + (">" if filled < bar_len else "") + " " * (bar_len - filled - (1 if filled < bar_len else 0))
    
    elapsed = time.time() - start_time
    if current_val > 0 and elapsed > 0:
        rate = current_val / elapsed
        rem_units = (total_val - current_val) / rate if rate > 0 else 0
        rem_str = f"Sisa ~{rem_units:.1f} dtk" if rem_units < 60 else f"Sisa ~{rem_units / 60:.1f} mnt"
        speed_str = f"{rate:.1f}x" if is_time else f"{rate:.1f} chunk/s"
    else:
        rem_str = "Menghitung..."
        speed_str = "--"
        
    if is_time:
        cur_str = format_timestamp(current_val)
        tot_str = format_timestamp(total_val)
        val_str = f"{cur_str} / {tot_str}"
    else:
        val_str = f"{int(current_val)}/{int(total_val)} chunks"
    
    sys.stdout.write(f"\r{prefix} : [{bar}] {pct:5.1f}% | {val_str} | Speed: {speed_str} | {rem_str}   ")
    sys.stdout.flush()

def finish_progress_bar(prefix: str, elapsed_time: float, extra: str = ""):
    """Menutup baris progres dengan status selesai"""
    bar_len = 25
    extra_str = f" {extra}" if extra else ""
    sys.stdout.write(f"\r{prefix} : [{'=' * bar_len}] 100.0% | Selesai ({elapsed_time:.1f} detik){extra_str}                                    \n")
    sys.stdout.flush()

def init_diarizer(num_speakers: int = 0, num_threads: int = 4):
    """Inisialisasi model Diarization (PyAnnote Segmentation 3.0 + 3D-Speaker ONNX)"""
    sys.stdout.write("Memuat bobot model Diarization... ")
    sys.stdout.flush()
    seg_model = hf_hub_download(
        repo_id="csukuangfj/sherpa-onnx-pyannote-segmentation-3-0",
        filename="model.onnx"
    )
    emb_model = hf_hub_download(
        repo_id="csukuangfj/speaker-embedding-models",
        filename="3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
    )

    clusters = num_speakers if num_speakers > 0 else -1
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=seg_model)
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=emb_model, num_threads=num_threads),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=clusters, threshold=0.5)
    )
    sd = sherpa_onnx.OfflineSpeakerDiarization(config)
    print("Siap.")
    return sd

def get_speaker_for_time(t: float, diar_segs) -> str:
    """Mencari pembicara yang aktif pada detik ke-t"""
    for seg in diar_segs:
        if seg.start <= t <= seg.end:
            return f"Pembicara {seg.speaker + 1}"
    min_dist = float("inf")
    closest_spk = "Pembicara 1"
    for seg in diar_segs:
        dist = min(abs(t - seg.start), abs(t - seg.end))
        if dist < min_dist:
            min_dist = dist
            closest_spk = f"Pembicara {seg.speaker + 1}"
    return closest_spk

def transcribe_audio(
    audio_path: str,
    model_size: str = "small",
    language: str = "id",
    beam_size: int = 1,
    cpu_threads: int = 4,
    use_vad: bool = True,
    device: str = "cpu",
    output_file: str = None,
    reduce_noise: bool = True,
    noise_decrease: float = 0.75,
    save_cleaned_audio: bool = False,
    diarize: bool = True,
    num_speakers: int = 0,
    show_text: bool = False
):
    if not os.path.exists(audio_path):
        print(f"Error: File audio '{audio_path}' tidak ditemukan.")
        print("Penggunaan: python transcribe.py [path_file_audio]")
        return

    base_name = os.path.splitext(audio_path)[0]
    if not output_file:
        output_file = f"{base_name}_transkripsi.txt"

    compute_type = "float16" if device == "cuda" else "int8"

    print("=" * 68)
    print("      Speech to Text & Diarization Rapat (Optimized Fast Mode)      ")
    print("=" * 68)
    print(f"File Audio       : {audio_path}")
    print(f"Ukuran Model     : {model_size.upper()}")
    print(f"Mode Pencarian   : Greedy Search (beam_size={beam_size}) [Cepat & Hemat CPU]")
    print(f"Alokasi CPU      : {cpu_threads} Cores (Multi-Thread)")
    print(f"Bahasa Target    : {language if language else 'Otomatis (Auto-detect)'}")
    print(f"Device           : {device.upper()} (compute_type='{compute_type}')")
    print(f"Noise Reduction  : {'Aktif (' + str(int(noise_decrease*100)) + '%)' if reduce_noise else 'Non-aktif'}")
    print(f"Filter Hening    : {'Aktif (Silero VAD)' if use_vad else 'Non-aktif'}")
    print(f"Diarization      : {'Aktif (Auto-detect)' if (diarize and num_speakers == 0) else ('Aktif (' + str(num_speakers) + ' pembicara)' if diarize else 'Non-aktif')}")
    print(f"File Output Teks : {output_file} (Ditulis langsung ke file)")
    print("-" * 68)

    total_start = time.time()

    # Buka dan siapkan file output teks sejak awal
    with open(output_file, "w", encoding="utf-8") as f_out:
        f_out.write(f"Notulensi Transkripsi Rapat: {audio_path}\n")
        f_out.write(f"Model: {model_size} | Beam Size: {beam_size} | CPU Cores: {cpu_threads}\n")
        f_out.write(f"Diarization: {'Aktif' if diarize else 'Non-aktif'}\n")
        f_out.write(f"Noise Reduction: {'Aktif' if reduce_noise else 'Non-aktif'}\n")
        f_out.write("=" * 68 + "\n\n")
        f_out.flush()

        # Baca audio
        sys.stdout.write("Membaca file audio... ")
        sys.stdout.flush()
        raw_audio = decode_audio(audio_path, sampling_rate=16000)
        audio_processed = raw_audio
        audio_duration_sec = len(raw_audio) / 16000.0
        print(f"Selesai (Durasi: {format_timestamp(audio_duration_sec)}).")

        # Tahap 1: Reduksi Derau (Noise Reduction) dengan Progress Bar
        if reduce_noise:
            nr_start = time.time()
            chunk_sec = 30
            chunk_samples = 16000 * chunk_sec
            noise_sample = raw_audio[:min(len(raw_audio), 16000)]
            cleaned_parts = []

            for start_sample in range(0, len(raw_audio), chunk_samples):
                end_sample = min(start_sample + chunk_samples, len(raw_audio))
                chunk = raw_audio[start_sample:end_sample]
                cleaned_chunk = nr.reduce_noise(
                    y=chunk,
                    sr=16000,
                    y_noise=noise_sample,
                    prop_decrease=noise_decrease,
                    stationary=True
                )
                cleaned_parts.append(cleaned_chunk)
                cur_sec = end_sample / 16000.0
                render_progress_bar(
                    cur_sec,
                    audio_duration_sec,
                    nr_start,
                    prefix="[Tahap 1/3] Noise Reduction"
                )

            audio_processed = np.concatenate(cleaned_parts)
            nr_elapsed = time.time() - nr_start
            finish_progress_bar("[Tahap 1/3] Noise Reduction", nr_elapsed)

            if save_cleaned_audio:
                clean_audio_path = f"{base_name}_jernih.wav"
                sf.write(clean_audio_path, audio_processed, 16000)
                print(f"-> Salinan audio jernih disimpan ke: {clean_audio_path}")

        # Tahap 2: Speaker Diarization dengan Progress Bar
        diar_segs = None
        if diarize:
            sd = init_diarizer(num_speakers=num_speakers, num_threads=cpu_threads)
            d_start = time.time()

            def diar_callback(processed_chunks: int, total_chunks: int):
                render_progress_bar(
                    processed_chunks,
                    total_chunks,
                    d_start,
                    prefix="[Tahap 2/3] Diarization    ",
                    is_time=False
                )
                return 0

            diar_res = sd.process(audio_processed, callback=diar_callback)
            diar_segs = diar_res.sort_by_start_time()
            d_elapsed = time.time() - d_start
            finish_progress_bar(
                "[Tahap 2/3] Diarization    ",
                d_elapsed,
                extra=f"- Terdeteksi {diar_res.num_speakers} pembicara"
            )

        # Tahap 3: Transkripsi Whisper dengan Progress Bar
        step_num = "3/3" if diarize else "2/2"
        prefix_tag = f"[Tahap {step_num}] Transkripsi AI  "

        sys.stdout.write(f"Memuat model Whisper '{model_size}'... ")
        sys.stdout.flush()
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=1
        )
        print("Siap.")

        whisper_start = time.time()
        segments, info = model.transcribe(
            audio_processed,
            beam_size=beam_size,
            language=language,
            word_timestamps=True if diarize else False,
            vad_filter=use_vad,
            vad_parameters=dict(min_silence_duration_ms=500) if use_vad else None
        )

        total_lines_count = 0

        # Jika Diarization Aktif, kelompokkan percakapan per pembicara
        if diarize and diar_segs:
            current_speaker = None
            current_start = 0.0
            current_end = 0.0
            current_words = []

            for seg in segments:
                # Update progress bar terminal
                render_progress_bar(seg.end, audio_duration_sec, whisper_start, prefix=prefix_tag)

                if seg.words:
                    for w in seg.words:
                        mid = (w.start + w.end) / 2
                        spk = get_speaker_for_time(mid, diar_segs)
                        if spk != current_speaker:
                            if current_words:
                                time_str = f"[{format_timestamp(current_start)} -> {format_timestamp(current_end)}]"
                                line = f"{time_str} [{current_speaker}]: {' '.join(current_words)}"
                                f_out.write(line + "\n")
                                f_out.flush()
                                total_lines_count += 1
                                if show_text:
                                    print(f"\n{line}")
                            current_speaker = spk
                            current_start = w.start
                            current_words = [w.word.strip()]
                        else:
                            current_words.append(w.word.strip())
                        current_end = w.end
                else:
                    spk = get_speaker_for_time((seg.start + seg.end) / 2, diar_segs)
                    time_str = f"[{format_timestamp(seg.start)} -> {format_timestamp(seg.end)}]"
                    line = f"{time_str} [{spk}]: {seg.text.strip()}"
                    f_out.write(line + "\n")
                    f_out.flush()
                    total_lines_count += 1
                    if show_text:
                        print(f"\n{line}")

            if current_words:
                time_str = f"[{format_timestamp(current_start)} -> {format_timestamp(current_end)}]"
                line = f"{time_str} [{current_speaker}]: {' '.join(current_words)}"
                f_out.write(line + "\n")
                f_out.flush()
                total_lines_count += 1
                if show_text:
                    print(f"\n{line}")
        else:
            # Tanpa diarization
            for seg in segments:
                render_progress_bar(seg.end, audio_duration_sec, whisper_start, prefix=prefix_tag)
                time_str = f"[{format_timestamp(seg.start)} -> {format_timestamp(seg.end)}]"
                line = f"{time_str} {seg.text.strip()}"
                f_out.write(line + "\n")
                f_out.flush()
                total_lines_count += 1
                if show_text:
                    print(f"\n{line}")

        whisper_elapsed = time.time() - whisper_start
        finish_progress_bar(prefix_tag, whisper_elapsed)

    total_time = time.time() - total_start
    whisper_time = time.time() - whisper_start
    speed_factor = (audio_duration_sec / whisper_time) if whisper_time > 0 else 0

    print("-" * 68)
    print(f"RINGKASAN HASIL:")
    print(f"* Total Baris Kalimat Terdata  : {total_lines_count} baris dialog")
    print(f"* Waktu Transkripsi AI Saja    : {whisper_time:.1f} detik ({whisper_time/60:.1f} menit)")
    print(f"* Kecepatan Pemrosesan         : {speed_factor:.1f}x lebih cepat dari durasi audio asli")
    print(f"* Total Waktu Keseluruhan      : {total_time:.1f} detik ({total_time/60:.1f} menit)")
    print(f"* Hasil Tersimpan Langsung di  : {output_file}")
    print("=" * 68)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speech to Text & Diarization Rapat Cepat (Optimized).")
    parser.add_argument("audio", nargs="?", default="meeting_sample.wav", help="Path file audio (default: meeting_sample.wav)")
    parser.add_argument("--model", default="small", help="Pilihan model Whisper: tiny, base, small, medium, large-v3 (default: small)")
    parser.add_argument("--beam-size", type=int, default=1, help="Ukuran beam search (default: 1 untuk mode super cepat & hemat CPU)")
    parser.add_argument("--cpu-threads", type=int, default=4, help="Jumlah thread CPU (default: 4 core)")
    parser.add_argument("--language", default="id", help="Kode bahasa (default: id untuk Bahasa Indonesia)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Perangkat: cpu atau cuda (default: cpu)")
    parser.add_argument("--num-speakers", type=int, default=0, help="Jumlah pembicara jika diketahui (contoh: 2, 3). Default: 0 (otomatis)")
    parser.add_argument("--no-diarize", action="store_true", help="Nonaktifkan deteksi pembicara (diarization)")
    parser.add_argument("--no-vad", action="store_true", help="Nonaktifkan filter hening (VAD)")
    parser.add_argument("--no-noise-reduction", action="store_true", help="Nonaktifkan reduksi derau")
    parser.add_argument("--save-clean-audio", action="store_true", help="Simpan salinan audio yang telah dibersihkan (.wav)")
    parser.add_argument("--show-text", action="store_true", help="Tampilkan teks percakapan di terminal (default: False/langsung simpan file)")
    parser.add_argument("--output", default=None, help="Nama file tujuan penyimpanan teks (.txt)")
    
    args = parser.parse_args()
    transcribe_audio(
        args.audio,
        model_size=args.model,
        language=args.language,
        beam_size=args.beam_size,
        cpu_threads=args.cpu_threads,
        use_vad=not args.no_vad,
        device=args.device,
        output_file=args.output,
        reduce_noise=not args.no_noise_reduction,
        save_cleaned_audio=args.save_clean_audio,
        diarize=not args.no_diarize,
        num_speakers=args.num_speakers,
        show_text=args.show_text
    )





