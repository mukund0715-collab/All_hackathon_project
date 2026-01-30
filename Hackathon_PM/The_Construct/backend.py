import numpy as np
import platform

class ConstructMath:
    def __init__(self):
        self.os = platform.system()
        if self.os == "Windows":
            import cupy as cp
            self.xp = cp # Use CUDA
        elif self.os == "Darwin":
            # Apple M4: NumPy linked to Accelerate is often faster for small matrices
            self.xp = np 
            
    def cosine_similarity_batch(self, A, B):
        # Optimized broadcasted dot product
        norm_a = self.xp.linalg.norm(A, axis=1, keepdims=True)
        norm_b = self.xp.linalg.norm(B, axis=1, keepdims=True)
        return self.xp.dot(A, B.T) / (norm_a * norm_b.T)
