import time
from capture import get_youtube_audio_url, capture_audio
from compare import compare_audio

YOUTUBE_ID = "xrcMRK9QVLM"
WEBSITE_STREAM_URL = "https://aajtaklive-amd.akamaized.net/hls/live/2014416/aajtak/aajtaklive/live_360p/chunks.m3u8"

YT_FILE = "temp/yt.wav"
WEB_FILE = "temp/web.wav"

def run():
    print("Resolving YouTube stream...")
    yt_audio_url = get_youtube_audio_url(YOUTUBE_ID)

    print("Capturing audio chunks...")
    capture_audio(yt_audio_url, YT_FILE)
    capture_audio(WEBSITE_STREAM_URL, WEB_FILE)

    print("Comparing audio...")
    score = compare_audio(YT_FILE, WEB_FILE)

    print(f"Similarity Score: {score:.3f}")

    if score > 0.90:
        print("✅ Same audio stream")
    elif score > 0.70:
        print("⚠️ Same audio but delayed / recompressed")
    else:
        print("❌ Different audio")

if __name__ == "__main__":
    run()
