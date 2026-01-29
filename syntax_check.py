
import sys
import os

print("Checking syntax...")
try:
    import sharp.utils.spherical
    print("spherical.py imported successfully")
except Exception as e:
    print(f"Error importing spherical.py: {e}")

try:
    # mock comfy imports
    sys.modules['folder_paths'] = type('obj', (object,), {'get_output_directory': lambda: '.'})
    import nodes.spag_predict
    print("spag_predict.py imported successfully")
except Exception as e:
    print(f"Error importing spag_predict.py: {e}")
