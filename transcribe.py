import os
import subprocess
from tqdm import tqdm
import yt_dlp
from faster_whisper import WhisperModel

# -------------------- Audio Download --------------------
def download_audio(video_url, output_dir="downloads"):
    """Downloads audio track only from video using yt-dlp"""
    print(f"[INFO] Downloading audio from: {video_url}")
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
        }]
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        filename = ydl.prepare_filename(info)
        audio_file = os.path.splitext(filename)[0] + ".mp3"

    print(f"[INFO] Audio saved to: {audio_file}")
    return audio_file

# -------------------- Audio Splitting --------------------
def split_audio_ffmpeg(audio_file, chunk_length_sec=300):
    """Split audio into chunks (~5 min) using ffmpeg"""
    chunk_dir = "chunks"
    os.makedirs(chunk_dir, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-i", audio_file,
        "-f", "segment",
        "-segment_time", str(chunk_length_sec),
        "-c", "copy",
        os.path.join(chunk_dir, "chunk%03d.wav")
    ]

    print(f"[INFO] Splitting audio into ~{chunk_length_sec//60} min chunks...")
    subprocess.run(cmd, check=True)
    chunk_files = sorted([os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir)])
    print(f"[INFO] Created {len(chunk_files)} chunk(s).")
    return chunk_files

# -------------------- Sequential Transcription with Loader --------------------
def transcribe_audio_sequential(audio_file, model_size="small", chunk_length_sec=300):
    """Transcribes audio sequentially with a progress bar for each chunk"""
    print(f"[INFO] Splitting audio into chunks...")
    chunks = split_audio_ffmpeg(audio_file, chunk_length_sec)
    print(f"[INFO] Transcribing {len(chunks)} chunk(s)...")

    txt_file = os.path.splitext(audio_file)[0] + ".txt"

    with open(txt_file, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(tqdm(chunks, desc="Transcribing chunks")):
            print(f"[INFO] Transcribing chunk {i+1}/{len(chunks)}: {chunk}")
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
            segments, _ = model.transcribe(chunk)
            for segment in segments:
                f.write(segment.text.strip() + " ")
            os.remove(chunk)
            print(f"[INFO] Finished chunk {i+1}/{len(chunks)} and deleted temporary file.")

    os.remove(audio_file)
    print(f"[INFO] Transcription complete. Saved to: {txt_file}")
    return txt_file

# -------------------- URL Processing --------------------
def process_url(url, model_size="small"):
    """End-to-end: download -> split -> transcribe sequentially; handles Rumble specially"""
    print(f"[INFO] Processing URL: {url}")

    # For Rumble, use yt-dlp directly
    if "rumble.com" in url:
        print("[INFO] Detected Rumble URL, using yt-dlp directly.")
        audio_file = download_audio(url)
        txt_file = transcribe_audio_sequential(audio_file, model_size=model_size)
        print(f"[INFO] Transcript saved to {txt_file}")
    else:
        # For other sites, attempt to download directly from the URL
        audio_file = download_audio(url)
        txt_file = transcribe_audio_sequential(audio_file, model_size=model_size)
        print(f"[INFO] Transcript saved to {txt_file}")

# -------------------- Example Usage --------------------
if __name__ == "__main__":
    test_url = "https://rumble.com/v6z2z5s-debunking-the-groyper-assassin-conspiracy.html"
    process_url(test_url)

