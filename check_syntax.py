
import sys
import os

print("Checking syntax...")
# Add parent dir to path to simulate package structure
parent_dir = os.path.dirname(os.getcwd())
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    # We need to simulate being inside the package 'ComfyUI-Sharp.nodes'
    # But usually custom nodes are loaded as modules.
    # Simple check: can we verify spag_predict syntax?
    pass
except Exception as e:
    pass

# We can try to compile the file to check for syntax errors
import py_compile
try:
    py_compile.compile('nodes/spag_predict.py', doraise=True)
    print("spag_predict.py syntax is valid")
except Exception as e:
    print(f"spag_predict.py syntax error: {e}")

try:
    py_compile.compile('sharp/utils/spherical.py', doraise=True)
    print("spherical.py syntax is valid")
except Exception as e:
    print(f"spherical.py syntax error: {e}")
