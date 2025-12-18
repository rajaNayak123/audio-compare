import os
import time
from datetime import datetime

from capture import get_youtube_audio_url, capture_audio
from compare import compare_audio


# === CONFIG ===
YOUTUBE_ID = "Nq2wYlWFucg"
WEBSITE_STREAM_URL = "https://feeds.intoday.in/aajtak/api/master.m3u8"

YT_FILE = "temp/yt.wav"
WEB_FILE = "temp/web.wav"

CHUNK_SECONDS = 20
SLEEP_BETWEEN = 3
NUM_CHUNKS_AGG = 5

# Adjusted thresholds for more lenient matching
SAME_THRESHOLD = 0.50  # Lowered from 0.70
SIMILAR_THRESHOLD = 0.35  # Lowered from 0.45


def ensure_temp_dir():
    os.makedirs(os.path.dirname(YT_FILE) or ".", exist_ok=True)


def file_ok(path: str, min_bytes: int = 1000) -> bool:
    return os.path.exists(path) and os.path.getsize(path) >= min_bytes


def run():
    ensure_temp_dir()

    print("\n" + "=" * 75)
    print("AUDIO STREAM COMPARISON - WITH TIME OFFSET DETECTION")
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
        print(f"✗ ERROR: {e}\n")
        return

    chunk_idx = 0
    similarities = []
    offsets = []
    offset_confidences = []

    try:
        while True:
            chunk_idx += 1
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            print(f"\n{'='*75}")
            print(f"[{timestamp}] CHUNK {chunk_idx}")
            print(f"{'='*75}")
            print("📥 Capturing audio from both sources...")

            # Capture both streams
            try:
                capture_audio(yt_audio_url, YT_FILE, duration=CHUNK_SECONDS, source_name="YouTube")
                print("✓ YouTube audio captured")
            except Exception as e:
                print(f"✗ YouTube capture failed: {e}")
                time.sleep(SLEEP_BETWEEN)
                continue

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

            # Compare with offset detection
            print("🔎 Comparing with time offset detection...")
            try:
                similarity, offset, offset_conf = compare_audio(YT_FILE, WEB_FILE, detect_offset=True)
                
                similarities.append(similarity)
                offsets.append(offset)
                offset_confidences.append(offset_conf)
                
            except Exception as e:
                print(f"✗ Comparison error: {e}")
                time.sleep(SLEEP_BETWEEN)
                continue

            # Display results
            print(f"\n📊 Chunk {chunk_idx} Results:")
            print(f"   Similarity: {similarity:.3f}")
            print(f"   Time Offset: {offset:+.2f}s (confidence: {offset_conf:.2f})")
            
            if similarity >= SAME_THRESHOLD:
                print(f"   Verdict: ✅ SAME AUDIO STREAM")
            elif similarity >= SIMILAR_THRESHOLD:
                print(f"   Verdict: ⚠️  SIMILAR AUDIO (compression/delay)")
            else:
                print(f"   Verdict: ❌ DIFFERENT AUDIO")

            # Aggregate results
            if len(similarities) >= NUM_CHUNKS_AGG:
                last_sim = similarities[-NUM_CHUNKS_AGG:]
                last_off = offsets[-NUM_CHUNKS_AGG:]
                last_conf = offset_confidences[-NUM_CHUNKS_AGG:]
                
                avg_sim = sum(last_sim) / len(last_sim)
                avg_off = sum(last_off) / len(last_off)
                avg_conf = sum(last_conf) / len(last_conf)
                high_matches = sum(1 for s in last_sim if s >= SAME_THRESHOLD)

                print(f"\n{'─'*75}")
                print(f"📈 AGGREGATE (Last {NUM_CHUNKS_AGG} chunks)")
                print(f"{'─'*75}")
                print(f"Average Similarity: {avg_sim:.3f}")
                print(f"Average Time Offset: {avg_off:+.2f}s (avg confidence: {avg_conf:.2f})")
                print(f"Matches above {SAME_THRESHOLD}: {high_matches}/{NUM_CHUNKS_AGG}")

                if high_matches >= NUM_CHUNKS_AGG - 1:
                    print(f"\n🎯 VERDICT: ✅✅ SAME AUDIO STREAM")
                    if abs(avg_off) > 2.0 and avg_conf > 0.5:
                        print(f"   ⚠️  NOTE: Streams have ~{abs(avg_off):.1f}s offset (being auto-corrected)")
                elif avg_sim >= SIMILAR_THRESHOLD:
                    print(f"\n🎯 VERDICT: ⚠️ SIMILAR AUDIO STREAMS")
                else:
                    print(f"\n🎯 VERDICT: ❌ DIFFERENT AUDIO STREAMS")
                print(f"{'─'*75}")

            print(f"⏳ Waiting {SLEEP_BETWEEN}s...")
            time.sleep(SLEEP_BETWEEN)

    except KeyboardInterrupt:
        print("\n\n⏹ Stopped by user")
        if similarities:
            avg_sim = sum(similarities) / len(similarities)
            avg_off = sum(offsets) / len(offsets)
            print(f"\nFinal Statistics ({len(similarities)} chunks):")
            print(f"  Average Similarity: {avg_sim:.3f}")
            print(f"  Average Offset: {avg_off:+.2f}s")


if __name__ == "__main__":
    run()
    