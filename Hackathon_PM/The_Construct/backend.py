import sys
import platform

# --- HARDWARE SELECTOR ---
OS_TYPE = platform.system()

try:
    if OS_TYPE == "Windows":
        import cupy as np
        print("[BACKEND] Running on NVIDIA GPU (CuPy)")
    elif OS_TYPE == "Darwin":
        # Mac M-chips are fast with NumPy (Accelerate) for small vectors
        # or we can use torch.mps if needed later. For now, NumPy is safer.
        import numpy as np
        print("[BACKEND] Running on Apple Silicon (NumPy/Accelerate)")
    else:
        import numpy as np
        print("[BACKEND] Running on CPU (Standard NumPy)")
except ImportError:
    import numpy as np
    print("[WARN] Hardware acceleration missing. Falling back to CPU NumPy.")

# --- CORE MATH FUNCTIONS ---

def normalize_to_hip(landmarks):
    """
    Transforms raw MediaPipe landmarks to a Hip-Relative Coordinate System.
    Source: TDD Section 1.3.1
    """
    # Landmarks is a (33, 3) array of [x, y, z]
    
    # 1. Calculate Hip Center (Origin)
    # MediaPipe IDs: 23 (Left Hip), 24 (Right Hip)
    left_hip = landmarks[23]
    right_hip = landmarks[24]
    
    # Origin = (Left_Hip + Right_Hip) / 2
    hip_center = (left_hip + right_hip) / 2.0
    
    # 2. Subtract Origin from ALL points (Translation)
    normalized = landmarks - hip_center
    
    return normalized

def calculate_cosine_similarity(v1, v2):
    """
    Returns the cosine similarity between two vectors.
    Range: -1 (Opposite) to 1 (Perfect Match)
    Source: TDD Section 1.2
    """
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
        
    return np.dot(v1, v2) / (norm_v1 * norm_v2)