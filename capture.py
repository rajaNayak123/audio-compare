import subprocess
import os


def get_youtube_audio_url(youtube_id: str) -> str:
    """
    Resolve a playable stream URL for a YouTube live video.
    """
    cmd = [
        "yt-dlp",
        "-f", "95/94/93/92/91/best",
        "-g",
        f"https://www.youtube.com/watch?v={youtube_id}",
    ]
    try:
        url = subprocess.check_output(cmd, stderr=subprocess.PIPE).decode().strip()
        if not url:
            raise RuntimeError("yt-dlp returned empty URL")
        return url
    except Exception as e:
        raise RuntimeError(f"Failed to resolve YouTube stream: {e}")


def capture_audio(stream_url: str, output_file: str, duration: int = 20, source_name: str = "stream") -> None:
    """
    Capture audio from a stream URL into a WAV file.
    Video is dropped; audio is mono 16 kHz.
    """
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    
    # Headers for website streams (Akamai CDN requires this)
    headers = (
        "Referer: https://aajtak.in/ "
        "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    )
    
    cmd = [
        "ffmpeg",
        "-y",
        "-headers", headers,
        "-i", stream_url,
        "-t", str(duration),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        output_file,
    ]
    
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=60)
    
    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="ignore")[-200:]
        raise RuntimeError(f"{source_name} capture failed: {error}")
    
    # Verify file was created and has content
    if not os.path.exists(output_file) or os.path.getsize(output_file) < 1000:
        raise RuntimeError(f"{source_name} capture produced empty file")
