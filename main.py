import os
import time
import sys
from datetime import datetime

from capture import get_youtube_audio_url, capture_audio
from compare import compare_audio


# === CONFIG ===

# Replace with CURRENT live stream ID (check if it's still live!)
YOUTUBE_ID = "Nq2wYlWFucg"

# Website stream (Aaj Tak)
WEBSITE_STREAM_URL = "https://feeds.intoday.in/aajtak/api/master.m3u8"

YT_FILE = "temp/yt.wav"
WEB_FILE = "temp/web.wav"

CHUNK_SECONDS = 20
SLEEP_BETWEEN = 3
NUM_CHUNKS_AGG = 5

SAME_THRESHOLD = 0.75  # Lowered tolerance for real-world compression
SIMILAR_THRESHOLD = 0.45


def ensure_temp_dir():
    os.makedirs(os.path.dirname(YT_FILE) or ".", exist_ok=True)


def file_ok(path: str, min_bytes: int = 1000) -> bool:
    return os.path.exists(path) and os.path.getsize(path) >= min_bytes


def run():
    ensure_temp_dir()

    print("\n" + "=" * 75)
    print("AUDIO STREAM COMPARISON - CHROMAPRINT FINGERPRINTING")
    print("=" * 75)
    print(f"YouTube ID: {YOUTUBE_ID}")
    print(f"Website URL: {WEBSITE_STREAM_URL}")
    print(f"Chunk Duration: {CHUNK_SECONDS}s | Thresholds: Same={SAME_THRESHOLD}, Similar={SIMILAR_THRESHOLD}")
    print("=" * 75 + "\n")

    print("🔍 Resolving YouTube stream...")
    try:
        yt_audio_url = get_youtube_audio_url(YOUTUBE_ID)
        print(f"✓ YouTube URL resolved\n")
    except Exception as e:
        print(f"✗ ERROR: {e}")
        print("\n⚠️  YouTube live may have ended or the ID is wrong.")
        print(f"   Visit: https://www.youtube.com/@aajtak/live")
        print("   Get the new video ID and update YOUTUBE_ID in main.py\n")
        return

    chunk_idx = 0
    similarities = []
    prev_yt_fp = None
    yt_frozen = False

    try:
        while True:
            chunk_idx += 1
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            print(f"\n{'='*75}")
            print(f"[{timestamp}] CHUNK {chunk_idx}")
            print(f"{'='*75}")
            print("📥 Capturing audio from both sources...")

            # Capture YouTube
            try:
                capture_audio(yt_audio_url, YT_FILE, duration=CHUNK_SECONDS, source_name="YouTube")
                print("✓ YouTube audio captured")
            except Exception as e:
                print(f"✗ YouTube capture failed: {e}")
                time.sleep(SLEEP_BETWEEN)
                continue

            # Capture Website
            try:
                capture_audio(WEBSITE_STREAM_URL, WEB_FILE, duration=CHUNK_SECONDS, source_name="Website")
                print("✓ Website audio captured")
            except Exception as e:
                print(f"✗ Website capture failed: {e}")
                time.sleep(SLEEP_BETWEEN)
                continue

            if not file_ok(YT_FILE) or not file_ok(WEB_FILE):
                print("✗ Capture produced empty files")
                time.sleep(SLEEP_BETWEEN)
                continue

            # Compare
            print("🔎 Comparing fingerprints...")
            try:
                score = compare_audio(YT_FILE, WEB_FILE)
                similarities.append(score)
            except Exception as e:
                print(f"✗ Comparison error: {e}")
                time.sleep(SLEEP_BETWEEN)
                continue

            # Check for frozen YouTube stream
            from compare import find_fpcalc
            try:
                fpcalc = find_fpcalc()
                if fpcalc:
                    import subprocess
                    result = subprocess.run(
                        [fpcalc, "-raw", "-length", "120", YT_FILE],
                        capture_output=True, text=True, timeout=30
                    )
                    for line in result.stdout.split("\n"):
                        if line.startswith("FINGERPRINT="):
                            curr_yt_fp = line.split("=", 1)[1].strip()
                            if prev_yt_fp and curr_yt_fp == prev_yt_fp:
                                if not yt_frozen:
                                    print("\n⚠️  WARNING: YouTube fingerprint is frozen!")
                                    print("   YouTube stream is not advancing (may be buffering or offline)")
                                    yt_frozen = True
                            prev_yt_fp = curr_yt_fp
                            break
            except:
                pass

            # Display chunk result
            print(f"\n📊 Chunk {chunk_idx} Similarity: {score:.3f}")
            if score >= SAME_THRESHOLD:
                print("✅ Same audio stream")
            elif score >= SIMILAR_THRESHOLD:
                print("⚠️  Similar audio (compression/delay)")
            else:
                print("❌ Different audio")

            # Aggregate
            if len(similarities) >= NUM_CHUNKS_AGG:
                last = similarities[-NUM_CHUNKS_AGG:]
                avg = sum(last) / len(last)
                high = sum(1 for s in last if s >= SAME_THRESHOLD)

                print(f"\n{'─'*75}")
                print(f"📈 AGGREGATE (Last {NUM_CHUNKS_AGG} chunks)")
                print(f"{'─'*75}")
                print(f"Average: {avg:.3f} | Matches: {high}/{NUM_CHUNKS_AGG}")

                if high >= NUM_CHUNKS_AGG - 1:
                    print(f"\n🎯 VERDICT: ✅✅ SAME AUDIO STREAM (confirmed)")
                elif avg >= SIMILAR_THRESHOLD:
                    print(f"\n🎯 VERDICT: ⚠️ SIMILAR AUDIO (high compression/delay)")
                else:
                    print(f"\n🎯 VERDICT: ❌ DIFFERENT AUDIO STREAMS")
                print(f"{'─'*75}")

            print(f"⏳ Waiting {SLEEP_BETWEEN}s...")
            time.sleep(SLEEP_BETWEEN)

    except KeyboardInterrupt:
        print("\n\n⏹ Stopped by user")
        if similarities:
            avg = sum(similarities) / len(similarities)
            print(f"Final average: {avg:.3f} over {len(similarities)} chunks")


if __name__ == "__main__":
    run()
