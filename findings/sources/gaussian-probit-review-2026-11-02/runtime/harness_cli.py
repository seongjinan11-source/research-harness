"""Use the existing harness and installed YAML/Excel dependencies without bytecode."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
repo=Path(__file__).resolve().parents[2]/'research-harness'
sys.path.insert(0,str(repo/'src'))
sys.path.append(r'C:/Users/seongjin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages')
import harness
sys.exit(harness.main(sys.argv[1:]))
