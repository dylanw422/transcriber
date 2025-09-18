import os
import subprocess
from tqdm import tqdm
import yt_dlp
from faster_whisper import WhisperModel
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

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
def split_audio_ffmpeg(audio_file, chunk_length_sec=180):  # Reduced to 3min for better parallelization
    """Split audio into chunks (~3 min) using ffmpeg"""
    chunk_dir = "chunks"
    os.makedirs(chunk_dir, exist_ok=True)

    # Use base filename to avoid conflicts
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    chunk_pattern = os.path.join(chunk_dir, f"{base_name}_chunk%03d.wav")

    cmd = [
        "ffmpeg", "-y",  # Overwrite existing files
        "-i", audio_file,
        "-f", "segment",
        "-segment_time", str(chunk_length_sec),
        "-c", "copy",
        "-reset_timestamps", "1",  # Reset timestamps for each chunk
        chunk_pattern
    ]

    print(f"[INFO] Splitting audio into ~{chunk_length_sec//60} min chunks...")
    subprocess.run(cmd, check=True, capture_output=True)
    
    # Get chunk files for this specific audio file - ENSURE SORTED ORDER
    chunk_files = []
    for f in os.listdir(chunk_dir):
        if f.startswith(f"{base_name}_chunk") and f.endswith('.wav'):
            chunk_files.append(os.path.join(chunk_dir, f))
    
    # Sort by extracting chunk number more safely
    def extract_chunk_number(filename):
        try:
            # Extract number from pattern like "basename_chunk001.wav"
            basename = os.path.basename(filename)
            chunk_part = basename.split('_chunk')[1]  # Get part after "_chunk"
            number_part = chunk_part.split('.')[0]    # Remove extension
            return int(number_part)
        except (IndexError, ValueError):
            # Fallback to alphabetical sorting if pattern doesn't match
            return 0
    
    chunk_files.sort(key=extract_chunk_number)
    
    print(f"[INFO] Created {len(chunk_files)} chunk(s): {[os.path.basename(f) for f in chunk_files]}")
    return chunk_files

# -------------------- Global Model Instance --------------------
# Create a single model instance to reuse across all chunks
_whisper_model = None

def get_whisper_model(model_size="small"):
    """Get or create the global Whisper model instance optimized for M4"""
    global _whisper_model
    if _whisper_model is None:
        print(f"[INFO] Loading Whisper model: {model_size}")
        # Optimized settings for Mac M4
        _whisper_model = WhisperModel(
            model_size, 
            device="cpu",  # M4 CPU is very fast
            compute_type="int8",  # Good balance of speed/quality
            cpu_threads=0,  # Use all available cores
            num_workers=4   # Parallel processing within model
        )
        print(f"[INFO] Whisper model loaded successfully")
    return _whisper_model

# -------------------- Single Chunk Transcription --------------------
def transcribe_single_chunk(chunk_info):
    """Transcribe a single chunk - designed for parallel execution"""
    chunk_file, chunk_index, total_chunks = chunk_info
    
    try:
        model = get_whisper_model()
        segments, info = model.transcribe(
            chunk_file, 
            beam_size=1,  # Faster inference
            language="en",  # Specify language to skip detection
            condition_on_previous_text=False  # Faster for independent chunks
        )
        
        # Collect all text from segments
        chunk_text = " ".join([segment.text.strip() for segment in segments])
        
        # Clean up chunk file
        try:
            os.remove(chunk_file)
        except OSError:
            pass  # File might already be deleted
            
        print(f"[INFO] Completed chunk {chunk_index + 1}/{total_chunks}: {len(chunk_text)} chars")
        return chunk_index, chunk_text
        
    except Exception as e:
        print(f"[ERROR] Failed to transcribe chunk {chunk_index + 1}: {e}")
        try:
            os.remove(chunk_file)
        except OSError:
            pass
        return chunk_index, ""

# -------------------- Parallel Transcription --------------------
def transcribe_audio_parallel(audio_file, output_file, model_size="small", chunk_length_sec=180, max_workers=None):
    """Transcribes audio using parallel processing optimized for Mac M4"""
    
    # Determine optimal worker count for M4
    if max_workers is None:
        cpu_count = multiprocessing.cpu_count()
        # M4 has excellent single-core performance, so we can use more workers
        max_workers = min(cpu_count, 8)  # Cap at 8 to avoid memory issues
    
    print(f"[INFO] Using {max_workers} parallel workers for transcription")
    
    # Split audio into chunks
    chunks = split_audio_ffmpeg(audio_file, chunk_length_sec)
    
    # Prepare chunk info for parallel processing
    chunk_info_list = [(chunk, i, len(chunks)) for i, chunk in enumerate(chunks)]
    
    print(f"[INFO] Transcribing {len(chunks)} chunk(s) in parallel...")
    
    # Pre-load the model in main thread
    get_whisper_model(model_size)
    
    # Process chunks in parallel
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all chunks
        futures = {executor.submit(transcribe_single_chunk, chunk_info): chunk_info 
                  for chunk_info in chunk_info_list}
        
        # Collect results with progress bar
        for future in tqdm(futures, desc="Transcribing chunks"):
            try:
                chunk_index, chunk_text = future.result(timeout=600)  # 10 min timeout per chunk
                results[chunk_index] = chunk_text
            except Exception as e:
                print(f"[ERROR] Chunk failed: {e}")
    
    # Write results in correct chronological order
    print(f"[INFO] Writing transcription in chronological order...")
    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(len(chunks)):
            if i in results and results[i].strip():  # Only write non-empty chunks
                print(f"[DEBUG] Writing chunk {i}: {len(results[i])} chars")
                f.write(results[i] + " ")
            elif i not in results:
                print(f"[WARNING] Missing chunk {i}, continuing...")
                # Write a placeholder or skip
    
    print(f"[INFO] Final transcription contains {len(results)} chunks")
    
    # Cleanup
    try:
        os.remove(audio_file)
    except OSError:
        pass
    
    # Clean up any remaining chunk files
    chunk_dir = "chunks"
    if os.path.exists(chunk_dir):
        for file in os.listdir(chunk_dir):
            try:
                os.remove(os.path.join(chunk_dir, file))
            except OSError:
                pass
    
    print(f"[INFO] Transcription complete. Saved to: {output_file}")
    return output_file

# -------------------- Process URLs from File --------------------
def process_urls_from_file(txt_file_path, model_size="small", start=5, end=10, max_workers=None): # Start value should equal number of files in /transcripts/author folder
    """
    Process URLs in a txt file and save transcriptions in transcripts/{streamer}/{videoTitle}.txt.
    
    Parameters:
        txt_file_path (str): Path to the txt file containing URLs.
        model_size (str): Whisper model size.
        start (int): Start index (inclusive).
        end (int): End index (exclusive). If None, process until the end of the file.
        max_workers (int): Number of parallel workers for transcription.
    """
    streamer_name = os.path.splitext(os.path.basename(txt_file_path))[0]
    base_output_dir = os.path.join("transcripts", streamer_name)
    os.makedirs(base_output_dir, exist_ok=True)

    with open(txt_file_path, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    # Slice the URLs based on start/end
    urls_to_process = urls[start:end]

    print(f"[INFO] Processing URLs {start} to {start + len(urls_to_process)} of {len(urls)} total.")
    print(f"[INFO] Using model: {model_size}")

    for i, url in enumerate(urls_to_process, start=start):
        try:
            print(f"\n[INFO] Processing video {i+1-start}/{len(urls_to_process)}")
            audio_file, video_title = download_audio(url, output_dir="downloads")
            
            # Sanitize video title for filesystem
            safe_title = "".join(c for c in video_title if c.isalnum() or c in " _-").strip()
            transcript_file = os.path.join(base_output_dir, f"{safe_title}.txt")
            
            # Check if transcript already exists
            if os.path.exists(transcript_file):
                print(f"[INFO] Transcript already exists: {transcript_file}")
                try:
                    os.remove(audio_file)
                except OSError:
                    pass
                continue
            
            # Use parallel transcription
            transcribe_audio_parallel(audio_file, transcript_file, model_size=model_size, max_workers=max_workers)
            
        except Exception as e:
            print(f"[ERROR] Failed to process URL {i}: {url}\n{e}")

# -------------------- Performance Testing --------------------
def benchmark_transcription_methods(test_audio_file, model_size="small"):
    """Compare sequential vs parallel transcription performance"""
    import time
    
    print(f"[BENCHMARK] Testing transcription performance on: {test_audio_file}")
    
    # Test parallel method
    start_time = time.time()
    transcribe_audio_parallel(test_audio_file + "_copy1", "test_parallel.txt", model_size)
    parallel_time = time.time() - start_time
    
    print(f"[BENCHMARK] Parallel transcription time: {parallel_time:.2f} seconds")
    
    # Cleanup test files
    try:
        os.remove("test_parallel.txt")
    except OSError:
        pass

# -------------------- Example Usage --------------------
if __name__ == "__main__":
    txt_file = "videoURLs/nickjfuentes.txt"  # Replace with any .txt file in videoURLs
    
    # For Mac M4, these are optimized settings:
    # - model_size: "small" for speed, "medium" for better accuracy
    # - max_workers: None (auto-detect), or manually set to 4-8
    process_urls_from_file(
        txt_file, 
        model_size="small",  # or "medium" for better quality
        max_workers=6        # Optimal for M4
    )
