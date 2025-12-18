import os
import subprocess
import shutil
import librosa
import numpy as np
from scipy import signal


def find_fpcalc() -> str:
    """Find fpcalc binary on the system."""
    common_paths = [
        "/opt/homebrew/bin/fpcalc",
        "/usr/local/bin/fpcalc",
        "/usr/bin/fpcalc",
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    try:
        result = subprocess.run(["which", "fpcalc"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    
    return shutil.which("fpcalc")


def detect_time_offset(file1: str, file2: str, max_offset: int = 60) -> tuple:
    """
    Detect time offset between two audio files using cross-correlation.
    
    Args:
        file1: Path to first audio file (reference)
        file2: Path to second audio file (test)
        max_offset: Maximum offset to search in seconds
    
    Returns:
        (offset_seconds, confidence) where offset_seconds is how much file2 is delayed
    """
    try:
        print(f"\n  🔍 Detecting time offset...")
        
        # Load audio files
        y1, sr1 = librosa.load(file1, sr=16000, mono=True)
        y2, sr2 = librosa.load(file2, sr=16000, mono=True)
        
        if len(y1) == 0 or len(y2) == 0:
            print(f"  ⚠️  Empty audio file")
            return 0, 0.0
        
        # Extract onset-based features (more robust than raw audio)
        onset1 = librosa.onset.onset_strength(y=y1, sr=sr1)
        onset2 = librosa.onset.onset_strength(y=y2, sr=sr2)
        
        # Pad to same length
        max_len = max(len(onset1), len(onset2))
        onset1 = np.pad(onset1, (0, max_len - len(onset1)), mode='constant')
        onset2 = np.pad(onset2, (0, max_len - len(onset2)), mode='constant')
        
        # Normalize
        if np.std(onset1) > 0:
            onset1 = (onset1 - np.mean(onset1)) / np.std(onset1)
        if np.std(onset2) > 0:
            onset2 = (onset2 - np.mean(onset2)) / np.std(onset2)
        
        # Cross-correlate to find offset
        correlation = signal.correlate(onset1, onset2, mode='same')
        
        # Find the maximum correlation point
        max_idx = np.argmax(correlation)
        
        # Convert to seconds (hop_length=512 at sr=16000 = ~32ms per frame)
        hop_length = 512
        center = len(correlation) // 2
        offset_frames = max_idx - center
        offset_seconds = offset_frames * hop_length / sr1
        
        # Confidence: normalized correlation at peak
        max_corr = np.max(np.abs(correlation))
        if max_corr > 0:
            confidence = abs(correlation[max_idx]) / max_corr
        else:
            confidence = 0.0
        confidence = min(max(confidence, 0.0), 1.0)
        
        print(f"  ✓ Offset detected: {offset_seconds:.2f}s (confidence: {confidence:.2f})")
        print(f"    → Source 2 is {'AHEAD' if offset_seconds > 0 else 'BEHIND'} by {abs(offset_seconds):.2f}s")
        
        return offset_seconds, confidence
    
    except Exception as e:
        print(f"  ⚠️  Offset detection failed: {e}")
        return 0, 0.0


def align_audio_files(file1: str, file2: str, offset_seconds: float, output_file1: str, output_file2: str) -> bool:
    """
    Align both files by trimming the appropriate one based on offset.
    
    Returns True if alignment was successful, False otherwise.
    """
    try:
        print(f"  ✂️  Aligning audio (offset: {offset_seconds:.2f}s)...")
        
        os.makedirs(os.path.dirname(output_file1) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(output_file2) or ".", exist_ok=True)
        
        if abs(offset_seconds) < 0.5:
            # Offset too small, just copy both
            shutil.copy2(file1, output_file1)
            shutil.copy2(file2, output_file2)
            print(f"  ✓ Offset too small, using originals")
            return True
        
        if offset_seconds > 0:
            # file2 is ahead, skip first N seconds of file2, keep file1 as-is
            cmd1 = [
                "ffmpeg", "-y", "-i", file1,
                "-t", "20", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1", output_file1,
            ]
            cmd2 = [
                "ffmpeg", "-y", "-i", file2,
                "-ss", f"{abs(offset_seconds):.2f}",
                "-t", "20", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1", output_file2,
            ]
        else:
            # file2 is behind, skip first N seconds of file1, keep file2 as-is
            cmd1 = [
                "ffmpeg", "-y", "-i", file1,
                "-ss", f"{abs(offset_seconds):.2f}",
                "-t", "20", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1", output_file1,
            ]
            cmd2 = [
                "ffmpeg", "-y", "-i", file2,
                "-t", "20", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1", output_file2,
            ]
        
        result1 = subprocess.run(cmd1, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=30)
        result2 = subprocess.run(cmd2, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=30)
        
        if result1.returncode == 0 and result2.returncode == 0:
            print(f"  ✓ Both files aligned successfully")
            return True
        else:
            print(f"  ⚠️  Alignment failed")
            return False
    
    except Exception as e:
        print(f"  ⚠️  Alignment error: {e}")
        return False


def compare_audio(file1: str, file2: str, detect_offset: bool = True) -> tuple:
    """
    Compare two audio files with optional time offset detection.
    
    Returns:
        (similarity_score, offset_seconds, confidence)
    """
    if not os.path.exists(file1) or not os.path.exists(file2):
        raise FileNotFoundError(f"Audio file missing: {file1} or {file2}")
    
    fpcalc_path = find_fpcalc()
    if not fpcalc_path:
        raise RuntimeError("fpcalc not found. Install: brew install chromaprint")
    
    try:
        offset_seconds = 0.0
        offset_confidence = 0.0
        file1_to_compare = file1
        file2_to_compare = file2
        
        if detect_offset:
            offset_seconds, offset_confidence = detect_time_offset(file1, file2, max_offset=60)
            
            # If offset is significant, align the files
            if abs(offset_seconds) > 1.0:
                temp_aligned1 = "temp/aligned1.wav"
                temp_aligned2 = "temp/aligned2.wav"
                
                if align_audio_files(file1, file2, offset_seconds, temp_aligned1, temp_aligned2):
                    file1_to_compare = temp_aligned1
                    file2_to_compare = temp_aligned2
                    print(f"  → Comparing aligned files (offset corrected: {offset_seconds:.2f}s)")
        
        # Get fingerprints
        print(f"  Generating fingerprints...")
        fp1 = _get_fingerprint(fpcalc_path, file1_to_compare, "Source 1")
        if not fp1:
            return 0.0, offset_seconds, 0.0
        
        fp2 = _get_fingerprint(fpcalc_path, file2_to_compare, "Source 2")
        if not fp2:
            return 0.0, offset_seconds, 0.0
        
        # Compare fingerprints
        similarity = _compare_fingerprints_smart(fpcalc_path, fp1, fp2)
        
        return similarity, offset_seconds, offset_confidence
    
    except Exception as e:
        print(f"  Error during comparison: {e}")
        return 0.0, 0.0, 0.0


def _get_fingerprint(fpcalc_path: str, audio_file: str, label: str = "") -> str:
    """Generate chromaprint fingerprint."""
    try:
        result = subprocess.run(
            [fpcalc_path, "-raw", "-length", "120", audio_file],
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        if result.returncode != 0:
            print(f"  {label} fingerprint generation failed")
            return None
        
        for line in result.stdout.split("\n"):
            if line.startswith("FINGERPRINT="):
                fp = line.split("=", 1)[1].strip()
                if fp:
                    print(f"  ✓ {label} FP: {fp[:50]}... (len: {len(fp)})")
                    return fp
        
        return None
    
    except subprocess.TimeoutExpired:
        print(f"  {label} fingerprint generation timeout")
        return None


def _compare_fingerprints_smart(fpcalc_path: str, fp1: str, fp2: str) -> float:
    """
    Compare fingerprints using the best available method.
    """
    if not fp1 or not fp2 or len(fp1) < 10 or len(fp2) < 10:
        return 0.0
    
    # Check if fingerprints are identical
    if fp1 == fp2:
        print(f"  → Perfect fingerprint match!")
        return 1.0
    
    # Try parsing and comparing as integer arrays (chromaprint format)
    try:
        arr1 = [int(x) for x in fp1.split(',')]
        arr2 = [int(x) for x in fp2.split(',')]
        
        # Use sliding window comparison for offset-tolerant matching
        max_sim = 0.0
        window_size = min(len(arr1), len(arr2))
        
        # Try different alignments in fingerprint space
        for offset in range(-10, 11):
            if offset < 0:
                start1 = -offset
                start2 = 0
                length = min(len(arr1) - start1, len(arr2) - start2, window_size)
            else:
                start1 = 0
                start2 = offset
                length = min(len(arr1) - start1, len(arr2) - start2, window_size)
            
            if length <= 0:
                continue
            
            a1 = arr1[start1:start1 + length]
            a2 = arr2[start2:start2 + length]
            
            if len(a1) > 0 and len(a2) > 0:
                # Hamming distance on integer fingerprints
                # Allow up to 16-bit differences per hash (accounts for compression)
                matches = sum(1 for x, y in zip(a1, a2) if bin(x ^ y).count('1') <= 16)
                sim = matches / len(a1)
                max_sim = max(max_sim, sim)
        
        if max_sim > 0.4:
            print(f"  (fingerprint array match: {max_sim:.3f})")
            return max_sim
    except Exception as e:
        print(f"  (fingerprint parsing failed: {e})")
        pass
    
    # Try fpcalc's built-in compare
    try:
        result = subprocess.run(
            [fpcalc_path, "-compare", fp1, fp2],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            try:
                score = float(result.stdout.strip())
                normalized = min(max(score / 100.0, 0.0), 1.0)
                print(f"  (fpcalc-compare: {score:.1f}% → {normalized:.3f})")
                return normalized
            except:
                pass
    except:
        pass
    
    # Fallback: Check for fingerprint overlap
    overlap_score = _check_fingerprint_overlap(fp1, fp2)
    if overlap_score > 0.3:
        print(f"  (fingerprint overlap: {overlap_score:.3f})")
        return overlap_score
    
    # Last resort: Character-level similarity
    char_score = _fingerprint_char_similarity(fp1, fp2)
    print(f"  (character similarity: {char_score:.3f})")
    return char_score


def _check_fingerprint_overlap(fp1: str, fp2: str) -> float:
    """Check if fingerprint patterns overlap."""
    chunk_size = max(len(fp1), len(fp2)) // 8
    if chunk_size < 20:
        return 0.0
    
    matches = 0
    total_checks = 0
    
    for i in range(0, len(fp1) - chunk_size, chunk_size // 2):
        chunk1 = fp1[i : i + chunk_size]
        best_match = 0
        for j in range(0, len(fp2) - chunk_size, chunk_size // 2):
            chunk2 = fp2[j : j + chunk_size]
            if chunk1 == chunk2:
                best_match = 1.0
                break
            # Also check similarity
            elif len(chunk1) == len(chunk2):
                sim = sum(1 for a, b in zip(chunk1, chunk2) if a == b) / len(chunk1)
                best_match = max(best_match, sim)
        
        matches += best_match
        total_checks += 1
    
    if total_checks == 0:
        return 0.0
    
    return min(matches / total_checks, 1.0)


def _fingerprint_char_similarity(fp1: str, fp2: str) -> float:
    """Compute character-level similarity."""
    max_len = max(len(fp1), len(fp2))
    if max_len == 0:
        return 0.0
    
    fp1_padded = fp1.ljust(max_len, "0")
    fp2_padded = fp2.ljust(max_len, "0")
    
    matches = sum(1 for a, b in zip(fp1_padded, fp2_padded) if a == b)
    return matches / max_len
