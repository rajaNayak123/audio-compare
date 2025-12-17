import os
import subprocess
import shutil


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
    
    fpcalc = shutil.which("fpcalc")
    return fpcalc


def compare_audio(file1: str, file2: str) -> float:
    """
    Compare two audio files using fpcalc chromaprint fingerprinting.
    
    Returns:
        Float in [0, 1]. > 0.80 means same audio stream.
    """
    if not os.path.exists(file1) or not os.path.exists(file2):
        raise FileNotFoundError(f"Audio file missing: {file1} or {file2}")
    
    fpcalc_path = find_fpcalc()
    if not fpcalc_path:
        raise RuntimeError("fpcalc not found. Install: brew install chromaprint")
    
    try:
        # Get fingerprints with sufficient length for reliability
        fp1 = _get_fingerprint(fpcalc_path, file1, "Source 1")
        if not fp1:
            return 0.0
        
        fp2 = _get_fingerprint(fpcalc_path, file2, "Source 2")
        if not fp2:
            return 0.0
        
        # Compare using multiple methods
        score = _compare_fingerprints_smart(fpcalc_path, fp1, fp2)
        return float(score)
    
    except Exception as e:
        print(f"  Error during comparison: {e}")
        return 0.0


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
    
    # Check if fingerprints are identical (exact same audio)
    if fp1 == fp2:
        print(f"  → Perfect fingerprint match!")
        return 1.0
    
    # Try fpcalc's compare command
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
    
    # Fallback: Check for significant overlap (handles delays)
    overlap_score = _check_fingerprint_overlap(fp1, fp2)
    if overlap_score > 0.5:
        print(f"  (fingerprint overlap: {overlap_score:.3f})")
        return overlap_score
    
    # Fallback: Character-level similarity
    char_score = _fingerprint_char_similarity(fp1, fp2)
    print(f"  (character similarity: {char_score:.3f})")
    return char_score


def _check_fingerprint_overlap(fp1: str, fp2: str) -> float:
    """
    Check if fingerprint patterns overlap (handles time offsets).
    """
    # Split into small chunks and look for overlaps
    chunk_size = max(len(fp1), len(fp2)) // 8
    if chunk_size < 20:
        return 0.0
    
    matches = 0
    total_checks = 0
    
    # Slide fp2 through fp1 looking for matches
    for i in range(0, len(fp1) - chunk_size, chunk_size):
        chunk1 = fp1[i : i + chunk_size]
        for j in range(0, len(fp2) - chunk_size, chunk_size):
            chunk2 = fp2[j : j + chunk_size]
            if chunk1 == chunk2:
                matches += 1
        total_checks += 1
    
    if total_checks == 0:
        return 0.0
    
    return min(matches / total_checks, 1.0)


def _fingerprint_char_similarity(fp1: str, fp2: str) -> float:
    """
    Compute character-level similarity.
    """
    max_len = max(len(fp1), len(fp2))
    if max_len == 0:
        return 0.0
    
    fp1_padded = fp1.ljust(max_len, "0")
    fp2_padded = fp2.ljust(max_len, "0")
    
    matches = sum(1 for a, b in zip(fp1_padded, fp2_padded) if a == b)
    return matches / max_len
