import sys
import platform
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def bootstrap():
    os_type = platform.system()
    machine = platform.machine()
    
    print(f"[INIT] Detected OS: {os_type} | Architecture: {machine}")
    
    # 1. Base Dependencies
    print("[INIT] Installing Core Vision Stack...")
    common_pkgs = ["opencv-python", "mediapipe", "numpy", "rich", "colorama"]
    for pkg in common_pkgs:
        install(pkg)

    # 2. Hardware-Specific Math Stack
    if os_type == "Windows":
        print("[INIT] Windows Detected. Installing NVIDIA CUDA Stack...")
        # Installs CuPy for CUDA 12.x (Adjust if you have CUDA 11)
        try:
            install("cupy-cuda12x")
        except:
            print("[WARN] CuPy install failed. Ensure CUDA Toolkit is installed.")
            
        # Llama-cpp for NVIDIA
        # Note: Requires Visual Studio Build Tools C++
        print("[INIT] Installing Llama-cpp (CUDA)...")
        subprocess.run('set CMAKE_ARGS="-DGGML_CUDA=on" && pip install llama-cpp-python --force-reinstall --no-cache-dir', shell=True)

    elif os_type == "Darwin": # macOS
        print("[INIT] macOS Detected. Installing Apple Metal Stack...")
        # PyTorch with MPS support is standard on Mac
        install("torch")
        install("torchvision")
        
        # Llama-cpp for Apple Silicon (Metal)
        print("[INIT] Installing Llama-cpp (Metal)...")
        subprocess.run('CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir', shell=True)
        
    print("\n[SUCCESS] The Construct environment is ready.")

if __name__ == "__main__":
    bootstrap()