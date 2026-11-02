"""Hermetic entrypoint: imported historical projects also contain run.py."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import importlib.util
import runpy
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import common
spec = importlib.util.spec_from_file_location('run', root/'run.py')
module = importlib.util.module_from_spec(spec)
sys.modules['run'] = module
spec.loader.exec_module(module)
runpy.run_path(str(root/'analysis.py'), run_name='__main__')
