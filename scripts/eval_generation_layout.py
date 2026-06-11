"""Canonical alias: generation-level layout fidelity (grounding mIoU / SR@0.5 / CLIP-T).
Delegates to scripts/eval_layout.py."""
import runpy, sys, os
sys.argv[0] = os.path.join(os.path.dirname(__file__), "eval_layout.py")
runpy.run_path(sys.argv[0], run_name="__main__")
