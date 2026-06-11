"""Canonical alias: planner-level layout/depth metrics on VidOR test (baselines + ablations).
Delegates to scripts/eval_ablation.py (L1/GIoU/center/area/size_acc/overlap/depth_acc)."""
import runpy, sys, os
sys.argv[0] = os.path.join(os.path.dirname(__file__), "eval_ablation.py")
runpy.run_path(sys.argv[0], run_name="__main__")
