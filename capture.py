import subprocess
import os

def get_youtube_audio_url(youtube_id):
    cmd = [
        "yt-dlp",
        "-f", "95/94/93/92/91",  # pick highest available, fall back down
        "-g",
        f"https://www.youtube.com/watch?v={youtube_id}",
    ]
    return subprocess.check_output(cmd).decode().strip()



def capture_audio(stream_url, output_file, duration=10):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", stream_url,
        "-t", str(duration),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        output_file
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
