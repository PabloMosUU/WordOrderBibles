import sys
import os

# Add the src directory to Python path so we can import from wordorderbibles package
src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Add the scripts directory to Python path so we can import compression and nn_pasting
scripts_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts')
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

# Add the root directory to Python path so we can import word_pasting and word_splitting
root_path = os.path.dirname(os.path.dirname(__file__))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
