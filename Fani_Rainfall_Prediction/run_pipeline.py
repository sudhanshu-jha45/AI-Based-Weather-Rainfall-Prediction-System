import sys
import os

# Add the project root directory to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import run_end_to_end_pipeline

if __name__ == "__main__":
    print("Executing Fani Rainfall Prediction modular pipeline...")
    try:
        run_end_to_end_pipeline()
        print("\nPipeline run complete. Diagnostics and reports have been generated in outputs/ directory.")
    except Exception as e:
        print(f"\nFATAL: Pipeline run failed with exception:")
        import traceback
        traceback.print_exc()
        sys.exit(1)
