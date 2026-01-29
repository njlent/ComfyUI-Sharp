
import sys
import os

print("Checking syntax...")
# Add parent dir to path to simulate package structure
parent_dir = os.path.dirname(os.getcwd())
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    # Explicitly try to import the mappings from the file as __init__ does
    from nodes.spag_predict import NODE_CLASS_MAPPINGS
    print("Successfully imported NODE_CLASS_MAPPINGS from nodes.spag_predict")
except ImportError as e:
    print(f"FAILED to import NODE_CLASS_MAPPINGS: {e}")
except Exception as e:
    print(f"Other error during import: {e}")
