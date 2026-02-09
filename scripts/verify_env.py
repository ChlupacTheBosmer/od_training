import sys
import os

def check_optimizations():
    # Check for Mac-specific optimizations (MPS)
    try:
        import torch
        if torch.backends.mps.is_available():
            print("MPS (Metal Performance Shaders) is available! (Mac GPU acceleration)")
        else:
            print("MPS not available.")
            
        if torch.cuda.is_available():
            print("CUDA is available!")
        else:
            print("CUDA not available (Expected on Mac).")
            
    except ImportError:
        pass

def verify():
    print("--- Environment Verification ---")
    print(f"Python: {sys.version.split()[0]}")
    
    # Critical: Numpy
    try:
        import numpy
        print(f"Numpy: {numpy.__version__}")
        if numpy.__version__.startswith("2"):
            print("CRITICAL WARNING: Numpy 2.x detected! This may break rfdetr.")
    except ImportError as e:
        print(f"FAIL: Numpy not found: {e}")

    # Critical: Pydantic
    try:
        import pydantic
        print(f"Pydantic: {pydantic.VERSION}")
        if pydantic.VERSION.startswith("2"):
            print("WARNING: Pydantic 2.x detected via VERSION attribute (or check import).")
    except ImportError:
        try:
             import pydantic
             print(f"Pydantic: {pydantic.__version__}")
        except:
             print("FAIL: Pydantic not found")

    # Torch
    try:
        import torch
        print(f"Torch: {torch.__version__}")
    except ImportError as e:
        print(f"FAIL: Torch not found: {e}")

    # Models
    try:
        import ultralytics
        print(f"Ultralytics: {ultralytics.__version__}")
    except ImportError as e:
        print(f"FAIL: Ultralytics import failed: {e}")

    try:
        import rfdetr
        print("RF-DETR: Imported successfully")
    except ImportError as e:
        print(f"FAIL: RF-DETR import failed: {e}")
        
    try:
        import roboflow
        print("Roboflow: Imported successfully")
    except ImportError:
        print("FAIL: Roboflow import failed")

    check_optimizations()
    print("--- Verification Complete ---")

if __name__ == "__main__":
    verify()
