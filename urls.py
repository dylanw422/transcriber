import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
import time
import os

def get_rumble_videos(profile_url, max_pages=5, save_html=False):
    videos = []

    # Extract streamer name from URL path
    parsed = urlparse(profile_url)
    streamer = parsed.path.split("/c/")[-1].split("/")[0]

    for page_num in range(1, max_pages + 1):
        # Parse the URL and update the page parameter correctly
        query_params = parse_qs(parsed.query)
        query_params.pop("page", None)
        query_params["page"] = [str(page_num)]

        # Build new URL
        new_query = urlencode(query_params, doseq=True)
        page_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment))

        print(f"[INFO] Scraping page {page_num}: {page_url}")

        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
        }

        try:
            response = requests.get(page_url, headers=headers, timeout=15)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"[ERROR] Failed to fetch page {page_num}: {e}")
            break

        # Optionally save HTML for debugging
        if save_html:
            with open(f"{streamer}_page_{page_num}.html", "w", encoding="utf-8") as f:
                f.write(response.text)

        soup = BeautifulSoup(response.text, "html.parser")
        video_elements = soup.find_all("div", class_="videostream thumbnail__grid--item")

        if not video_elements:
            print(f"[INFO] No videos found on page {page_num}, stopping.")
            break

        for video in video_elements:
            link_tag = video.find("a", class_="videostream__link")
            if link_tag and link_tag.get("href"):
                video_url = "https://rumble.com" + link_tag["href"]
                # Skip URLs that contain 'ep.'
                if "ep." in video_url.lower():
                    continue
                if "america-first" in video_url.lower():
                    continue
                videos.append(video_url)

        time.sleep(2)

    # Save video URLs to a file named after the streamer
    os.makedirs("videoURLs", exist_ok=True)
    output_file = f"videoURLs/{streamer}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for url in videos:
            f.write(url + "\n")

    print(f"[INFO] Saved {len(videos)} video URLs to {output_file}")
    return videos

# Example usage
profile = "https://rumble.com/c/nickjfuentes?e9s=src_v1_cmd"
videos = get_rumble_videos(profile, max_pages=62, save_html=False)

