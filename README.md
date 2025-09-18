# Video Transcription Pipeline

This project is a Python-based pipeline for downloading, transcribing, and processing video content from the web. It's designed to be efficient and scalable, with a focus on parallel processing and optimized for Apple Silicon (M4) but configurable for other systems.

## Features

- **URL Scraping**: Scrapes video URLs from a Rumble channel.
- **Audio Downloading**: Downloads audio from video URLs using `yt-dlp`.
- **Audio Chunking**: Splits audio files into smaller chunks using `ffmpeg` for parallel processing.
- **Parallel Transcription**: Transcribes audio chunks in parallel using `faster-whisper` for significant speed improvements.
- **Optimized for Apple Silicon**: The transcription script is optimized for M4 CPUs, but can be configured for other architectures.
- **Organized Output**: Saves transcriptions in a structured directory format (`transcripts/<streamer>/<video_title>.txt`).

## How It Works

The pipeline consists of two main scripts:

1.  **`urls.py`**: This script scrapes video URLs from a given Rumble channel profile and saves them to a text file in the `videoURLs/` directory.
2.  **`transcribe.py`**: This script reads the URLs from the text file, and for each URL, it:
    1.  Downloads the audio-only version of the video.
    2.  Splits the audio into smaller, manageable chunks.
    3.  Transcribes each chunk in parallel using multiple CPU cores.
    4.  Combines the transcribed chunks into a single text file.
    5.  Cleans up the audio and chunk files after transcription.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install FFmpeg:**
    This project requires FFmpeg for audio manipulation. You can install it using Homebrew on macOS:
    ```bash
    brew install ffmpeg
    ```
    For other operating systems, please refer to the official FFmpeg website for installation instructions.

## Usage

### Step 1: Scrape Video URLs

1.  Open the `urls.py` file and set the `profile` variable to the Rumble channel URL you want to scrape.
2.  You can also adjust the `max_pages` variable to control how many pages of videos to scrape.
3.  Run the script:
    ```bash
    python urls.py
    ```
    This will create a file named `videoURLs/<streamer_name>.txt` containing the video URLs.

### Step 2: Transcribe the Videos

1.  Open the `transcribe.py` file.
2.  In the `if __name__ == "__main__":` block, set the `txt_file` variable to the path of the URL file you generated in the previous step.
3.  You can configure the transcription process by modifying the parameters in the `process_urls_from_file` function call:
    - `model_size`: The Whisper model to use (e.g., "tiny", "base", "small", "medium", "large"). "small" is a good balance of speed and accuracy.
    - `max_workers`: The number of parallel workers to use for transcription. The script is optimized to auto-detect the best setting for M4 chips, but you can manually set it.
4.  Run the script:
    ```bash
    python transcribe.py
    ```
    The script will download, process, and transcribe the videos, saving the transcriptions in the `transcripts/<streamer_name>/` directory.

## Dependencies

This project relies on the following main libraries:

-   `requests` & `beautifulsoup4`: For web scraping.
-   `yt-dlp`: For downloading video audio.
-   `ffmpeg-python`: For audio manipulation.
-   `faster-whisper`: For efficient audio transcription.
-   `torch`, `torchvision`, `torchaudio`: Required by `faster-whisper`.

For a full list of dependencies, see the `requirements.txt` file.

## Configuration

-   **Transcription Model**: You can change the `model_size` in `transcribe.py` to use different Whisper models. Larger models are more accurate but slower.
-   **Parallel Workers**: The `max_workers` parameter in `transcribe.py` can be tuned based on your CPU to optimize transcription speed.
-   **URL Scraping**: The `profile_url` and `max_pages` in `urls.py` can be changed to scrape different Rumble channels.
