#!/usr/bin/env python3
"""
Test: GraspGen installation, imports, submodule discovery, and basic functionality.
Run inside the Docker container with system Python 3.10.

Usage:
  python3 scripts/test_graspgen.py
  python3 scripts/test_graspgen.py --no-display   # skip visualization
"""
import sys
import os
import time
import importlib
import pkgutil
import numpy as np


# =========================================================================
# 1. GraspGen requirements.txt dependencies
# =========================================================================
def test_graspgen_requirements():
    """Test all packages from GraspGen's own requirements.txt."""
    print("[TEST] GraspGen requirements.txt dependencies...")
    results = {}

    # (import_name, pip_name, notes)
    modules = [
        ("h5py",             "h5py",              ""),
        ("hydra",            "hydra-core",         ""),
        ("matplotlib",       "matplotlib",         ""),
        ("meshcat",          "meshcat",            ""),
        ("numpy",            "numpy==1.26.4",      ""),
        ("webdataset",       "webdataset",         ""),
        ("sklearn",          "scikit-learn",        ""),
        ("scipy",            "scipy",              ""),
        ("tensorboard",      "tensorboard",         ""),
        ("trimesh",          "trimesh==4.5.3",      ""),
        ("transformers",     "transformers",        ""),
        ("tensordict",       "tensordict",          ""),
        ("diffusers",        "diffusers==0.11.1",   ""),
        ("timm",             "timm==1.0.15",        ""),
        ("huggingface_hub",  "huggingface-hub",     ""),
        ("OpenGL",           "PyOpenGL",            ""),
        ("addict",           "addict",              ""),
        ("spconv",           "spconv-cu126",        "replaced spconv-cu120"),
        ("yapf",             "yapf==0.40.1",        ""),
        ("tensorboardX",     "tensorboardx",        ""),
        ("torch_geometric",  "torch-geometric",     ""),
        ("yourdfpy",         "yourdfpy==0.0.56",    ""),
        ("imageio",          "imageio",             ""),
    ]

    for import_name, pip_name, notes in modules:
        try:
            mod = __import__(import_name)
            version = getattr(mod, "__version__", "")
            ver_str = f" ({version})" if version else ""
            note_str = f" [{notes}]" if notes else ""
            print(f"  [PASS] {import_name}{ver_str}{note_str}")
            results[import_name] = True
        except ImportError as e:
            print(f"  [FAIL] {import_name} (pip: {pip_name}) — {e}")
            results[import_name] = False

    # Packages removed intentionally
    print(f"  [SKIP] pickle5 (built-in since Python 3.8)")
    print(f"  [SKIP] sharedarray (not imported anywhere in GraspGen source code)")

    return results


# =========================================================================
# 2. Pipeline-specific dependencies (not in GraspGen's requirements.txt)
# =========================================================================
def test_pipeline_deps():
    """Test additional dependencies needed by our pipeline."""
    print("\n[TEST] Pipeline-specific dependencies...")
    results = {}

    modules = [
        ("torch",            "PyTorch"),
        ("torchvision",      "TorchVision"),

        ("pointnet2_ops",    "PointNet++ CUDA ops"),
        ("torch_scatter",    "torch-scatter"),
        ("torch_cluster",    "torch-cluster"),
        ("open3d",           "Open3D"),
        ("cv2",              "OpenCV"),
        ("omegaconf",        "OmegaConf"),
        ("pymodbus",         "PyModbus (gripper)"),
        ("serial",           "PySerial (gripper)"),
    ]

    for import_name, label in modules:
        try:
            mod = __import__(import_name)
            version = getattr(mod, "__version__", "")
            ver_str = f" ({version})" if version else ""
            print(f"  [PASS] {label}: {import_name}{ver_str}")
            results[import_name] = True
        except ImportError as e:
            print(f"  [FAIL] {label}: {import_name} — {e}")
            results[import_name] = False

    return results


# =========================================================================
# 3. GraspGen package deep inspection
# =========================================================================
def test_grasp_gen_imports():
    """Discover and test all grasp_gen submodules recursively."""
    print("\n[TEST] GraspGen package deep import test...")

    try:
        import grasp_gen
    except ImportError as e:
        print(f"  [FAIL] Cannot import grasp_gen: {e}")
        return {}, False

    pkg_dir = os.path.dirname(grasp_gen.__file__)
    print(f"  [PASS] grasp_gen package at: {pkg_dir}")

    results = {"grasp_gen": True}
    pass_count = 0
    fail_count = 0

    for importer, modname, ispkg in pkgutil.walk_packages(
        path=[pkg_dir], prefix="grasp_gen.", onerror=lambda x: None
    ):
        try:
            importlib.import_module(modname)
            print(f"  [PASS] {modname}")
            results[modname] = True
            pass_count += 1
        except Exception as e:
            err_msg = str(e)
            if len(err_msg) > 80:
                err_msg = err_msg[:77] + "..."
            print(f"  [WARN] {modname}: {err_msg}")
            results[modname] = False
            fail_count += 1

    print(f"\n  [INFO] GraspGen submodules: {pass_count} passed, {fail_count} warnings")
    return results, fail_count == 0


# =========================================================================
# 4. PyTorch + CUDA
# =========================================================================
def test_pytorch_cuda():
    """Test PyTorch CUDA integration."""
    print("\n[TEST] PyTorch CUDA...")
    import torch

    if not torch.cuda.is_available():
        print("  [FAIL] CUDA not available")
        return False

    gpu_name = torch.cuda.get_device_name(0)
    print(f"  [PASS] CUDA available: {gpu_name}")
    print(f"  [INFO] PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
    print(f"  [INFO] cuDNN {torch.backends.cudnn.version()}")

    t = torch.randn(256, 256, device="cuda")
    result = t @ t.T
    print(f"  [PASS] GPU matmul OK (256x256)")

    cc = torch.cuda.get_device_capability(0)
    print(f"  [INFO] Compute capability: {cc[0]}.{cc[1]}")

    return True


# =========================================================================
# 5. PointNet++ CUDA extensions
# =========================================================================
def test_pointnet2_cuda():
    """Test PointNet++ CUDA extensions with actual operations."""
    print("\n[TEST] PointNet++ CUDA extensions...")
    try:
        import torch
        import pointnet2_ops

        print(f"  [PASS] pointnet2_ops imported from: {pointnet2_ops.__file__}")

        ops = [a for a in dir(pointnet2_ops) if not a.startswith("_")]
        print(f"  [INFO] Available ops: {', '.join(ops[:10])}")

        if hasattr(pointnet2_ops, "pointnet2_utils"):
            utils = pointnet2_ops.pointnet2_utils
            try:
                pc = torch.randn(1, 128, 3, device="cuda")
                idx = utils.furthest_point_sample(pc, 32)
                print(f"  [PASS] furthest_point_sample: input (1,128,3) -> idx {idx.shape}")
            except Exception as e:
                print(f"  [WARN] furthest_point_sample test: {e}")

        return True
    except Exception as e:
        print(f"  [FAIL] PointNet++ error: {e}")
        return False


# =========================================================================
# 6. Model weights
# =========================================================================
def test_model_weights():
    """Check if GraspGen model weights are downloaded and inspect structure."""
    print("\n[TEST] GraspGen model weights...")
    weights_dir = "/opt/GraspGen/GraspGenModels"

    if not os.path.isdir(weights_dir):
        print(f"  [WARN] {weights_dir} not found")
        return False

    files = []
    configs = []
    checkpoints = []
    for root, dirs, fnames in os.walk(weights_dir):
        for f in fnames:
            fpath = os.path.join(root, f)
            size_mb = os.path.getsize(fpath) / 1e6
            rel = os.path.relpath(fpath, weights_dir)
            files.append((rel, size_mb))
            if f.endswith((".yml", ".yaml")):
                configs.append(rel)
            if f.endswith((".pt", ".pth", ".ckpt", ".bin", ".safetensors")):
                checkpoints.append((rel, size_mb))

    if not files:
        print(f"  [WARN] {weights_dir} is empty")
        return False

    total_mb = sum(s for _, s in files)
    print(f"  [PASS] {len(files)} files, {total_mb:.0f} MB total")

    if checkpoints:
        print(f"  [INFO] Checkpoints ({len(checkpoints)}):")
        for name, size in sorted(checkpoints):
            print(f"           {name} ({size:.1f} MB)")

    if configs:
        print(f"  [INFO] Gripper configs ({len(configs)}):")
        for name in sorted(configs):
            print(f"           {name}")

    return True


# =========================================================================
# 7. GPU memory
# =========================================================================
def test_gpu_memory():
    """Report GPU memory usage."""
    import torch
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        free = total - reserved
        print(f"\n[INFO] GPU Memory: {allocated:.2f} GB allocated, "
              f"{reserved:.2f} GB reserved, {free:.2f} GB free / {total:.1f} GB total")


# =========================================================================
# Main
# =========================================================================
def main():
    parser = __import__("argparse").ArgumentParser(description="Test GraspGen")
    parser.add_argument("--no-display", action="store_true", help="Skip visualization")
    args = parser.parse_args()

    print("=" * 60)
    print("  GraspGen Installation & Functionality Test (Python 3.10)")
    print("=" * 60)

    req_results = test_graspgen_requirements()
    pipe_results = test_pipeline_deps()
    gg_results, gg_all_ok = test_grasp_gen_imports()
    cuda_ok = test_pytorch_cuda()
    pn2_ok = test_pointnet2_cuda()
    weights_ok = test_model_weights()
    test_gpu_memory()

    # Summary
    req_pass = sum(1 for v in req_results.values() if v)
    req_fail = sum(1 for v in req_results.values() if not v)
    pipe_pass = sum(1 for v in pipe_results.values() if v)
    pipe_fail = sum(1 for v in pipe_results.values() if not v)
    gg_pass = sum(1 for v in gg_results.values() if v)
    gg_fail = sum(1 for v in gg_results.values() if not v)

    print(f"\n{'=' * 60}")
    print(f"  GraspGen deps:   {req_pass}/{req_pass + req_fail} passed")
    print(f"  Pipeline deps:   {pipe_pass}/{pipe_pass + pipe_fail} passed")
    print(f"  GraspGen modules:{gg_pass} passed, {gg_fail} warnings")
    print(f"  CUDA:            {'OK' if cuda_ok else 'FAIL'}")
    print(f"  PointNet++:      {'OK' if pn2_ok else 'FAIL'}")
    print(f"  Weights:         {'OK' if weights_ok else 'MISSING'}")
    print(f"{'=' * 60}")

    total_fail = req_fail + pipe_fail
    sys.exit(0 if total_fail == 0 and cuda_ok else 1)


if __name__ == "__main__":
    main()
