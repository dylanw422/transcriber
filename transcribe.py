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
        title = info.get("title", "transcription")

    print(f"[INFO] Audio saved to: {audio_file}")
    return audio_file, title

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

# -------------------- Sequential Transcription --------------------
def transcribe_audio_sequential(audio_file, output_file, model_size="small", chunk_length_sec=300):
    """Transcribes audio sequentially with a progress bar for each chunk"""
    print(f"[INFO] Splitting audio into chunks...")
    chunks = split_audio_ffmpeg(audio_file, chunk_length_sec)
    print(f"[INFO] Transcribing {len(chunks)} chunk(s)...")

    with open(output_file, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(tqdm(chunks, desc="Transcribing chunks")):
            print(f"[INFO] Transcribing chunk {i+1}/{len(chunks)}: {chunk}")
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
            segments, _ = model.transcribe(chunk)
            for segment in segments:
                f.write(segment.text.strip() + " ")
            os.remove(chunk)
            print(f"[INFO] Finished chunk {i+1}/{len(chunks)} and deleted temporary file.")

    os.remove(audio_file)
    print(f"[INFO] Transcription complete. Saved to: {output_file}")
    return output_file

# -------------------- Process URLs from File --------------------
def process_urls_from_file(txt_file_path, model_size="small", start=1, end=2): #start=0, end=None):
    """
    Process URLs in a txt file and save transcriptions in transcripts/{streamer}/{videoTitle}.txt.
    
    Parameters:
        txt_file_path (str): Path to the txt file containing URLs.
        model_size (str): Whisper model size.
        start (int): Start index (inclusive).
        end (int): End index (exclusive). If None, process until the end of the file.
    """
    streamer_name = os.path.splitext(os.path.basename(txt_file_path))[0]
    base_output_dir = os.path.join("transcripts", streamer_name)
    os.makedirs(base_output_dir, exist_ok=True)

    with open(txt_file_path, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    # Slice the URLs based on start/end
    urls_to_process = urls[start:end]

    print(f"[INFO] Processing URLs {start} to {start + len(urls_to_process)} of {len(urls)} total.")

    for i, url in enumerate(urls_to_process, start=start):
        try:
            audio_file, video_title = download_audio(url, output_dir="downloads")
            # Sanitize video title for filesystem
            safe_title = "".join(c for c in video_title if c.isalnum() or c in " _-").strip()
            transcript_file = os.path.join(base_output_dir, f"{safe_title}.txt")
            transcribe_audio_sequential(audio_file, transcript_file, model_size=model_size)
        except Exception as e:
            print(f"[ERROR] Failed to process URL {i}: {url}\n{e}")



# -------------------- Example Usage --------------------
if __name__ == "__main__":
    txt_file = "videoURLs/nickjfuentes.txt"  # Replace with any .txt file in videoURLs
    process_urls_from_file(txt_file, model_size="small")

