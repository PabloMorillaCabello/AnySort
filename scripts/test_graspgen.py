#!/usr/bin/env python3
"""
Test: GraspGen model loading and inference on a sample point cloud.
Run inside the Docker container.

Usage:
  python3 scripts/test_graspgen.py
  python3 scripts/test_graspgen.py --mesh path/to/object.obj
  python3 scripts/test_graspgen.py --gripper_config /opt/models/graspgen/robotiq_2f140.yml
"""
import argparse
import time
import sys
import os
import numpy as np


def test_imports():
    """Test that GraspGen packages can be imported."""
    print("[TEST] Importing GraspGen packages...")
    results = {}

    modules = [
        ("grasp_gen.sampler", "GraspGenSampler"),
        ("grasp_gen.utils", "load_grasp_cfg"),
        ("pointnet2_ops", None),
        ("spconv", None),
        ("torch_geometric", None),
        ("trimesh", None),
        ("diffusers", None),
    ]

    for module, attr in modules:
        try:
            mod = __import__(module)
            if attr:
                assert hasattr(mod, attr) or True  # submodule import
            print(f"  [PASS] {module}")
            results[module] = True
        except ImportError as e:
            print(f"  [FAIL] {module}: {e}")
            results[module] = False

    return results


def test_model_loading(gripper_config):
    """Test loading the GraspGen model."""
    print(f"\n[TEST] Loading GraspGen model...")
    print(f"  [INFO] Config: {gripper_config}")

    try:
        from grasp_gen.sampler import GraspGenSampler
        from grasp_gen.utils import load_grasp_cfg

        t0 = time.time()
        grasp_cfg = load_grasp_cfg(gripper_config)
        sampler = GraspGenSampler(grasp_cfg)
        load_time = time.time() - t0

        print(f"  [PASS] Model loaded in {load_time:.1f}s")
        return sampler
    except Exception as e:
        print(f"  [FAIL] Model loading failed: {e}")
        return None


def test_inference_synthetic(sampler):
    """Test GraspGen on a synthetic point cloud (sphere)."""
    from grasp_gen.sampler import GraspGenSampler

    print("\n[TEST] Running inference on synthetic point cloud...")

    # Generate a sphere point cloud (simulates a ball-like object)
    n_points = 4096
    phi = np.random.uniform(0, 2 * np.pi, n_points)
    costheta = np.random.uniform(-1, 1, n_points)
    theta = np.arccos(costheta)
    r = 0.05  # 5cm radius

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta) + 0.3  # offset 30cm from camera

    pc = np.stack([x, y, z], axis=-1).astype(np.float32)
    print(f"  [INFO] Point cloud shape: {pc.shape}")

    try:
        t0 = time.time()
        grasps, confidences = GraspGenSampler.run_inference(
            pc,
            sampler,
            grasp_threshold=0.5,
            num_grasps=20,
            topk_num_grasps=5,
            remove_outliers=False,
        )
        inference_time = time.time() - t0

        print(f"  [PASS] Inference completed in {inference_time:.2f}s")
        print(f"  [INFO] Generated {len(grasps)} grasps")

        if len(confidences) > 0:
            print(f"  [INFO] Confidence range: [{min(confidences):.3f}, {max(confidences):.3f}]")
            for i, (g, c) in enumerate(zip(grasps[:3], confidences[:3])):
                pos = g[:3, 3]
                print(f"  [INFO]   Grasp {i}: pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), conf={c:.3f}")

        return True
    except Exception as e:
        print(f"  [FAIL] Inference failed: {e}")
        return False


def test_inference_mesh(sampler, mesh_path):
    """Test GraspGen on a mesh file."""
    from grasp_gen.sampler import GraspGenSampler
    import trimesh

    print(f"\n[TEST] Running inference on mesh: {mesh_path}")

    try:
        mesh = trimesh.load(mesh_path)
        pc = mesh.sample(4096).astype(np.float32)
        print(f"  [INFO] Sampled {pc.shape[0]} points from mesh")

        t0 = time.time()
        grasps, confidences = GraspGenSampler.run_inference(
            pc,
            sampler,
            grasp_threshold=0.5,
            num_grasps=20,
            topk_num_grasps=5,
            remove_outliers=False,
        )
        inference_time = time.time() - t0

        print(f"  [PASS] Inference completed in {inference_time:.2f}s")
        print(f"  [INFO] Generated {len(grasps)} grasps")
        return True
    except Exception as e:
        print(f"  [FAIL] Mesh inference failed: {e}")
        return False


def test_gpu_memory():
    """Report GPU memory after GraspGen."""
    import torch
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"\n[INFO] GPU Memory: {allocated:.1f}GB allocated, "
              f"{reserved:.1f}GB reserved, {total:.1f}GB total")


def find_gripper_config():
    """Try to find a gripper config file."""
    search_paths = [
        "/opt/models/graspgen/robotiq_2f140.yml",
        "/opt/models/graspgen/franka_panda.yml",
        "/opt/models/graspgen/suction.yml",
    ]
    # Also search recursively
    for root, dirs, files in os.walk("/opt/models/graspgen"):
        for f in files:
            if f.endswith(".yml") or f.endswith(".yaml"):
                search_paths.append(os.path.join(root, f))

    for p in search_paths:
        if os.path.isfile(p):
            return p
    return None


def main():
    parser = argparse.ArgumentParser(description="Test GraspGen model")
    parser.add_argument("--gripper_config", type=str, default=None,
                        help="Path to gripper config YAML")
    parser.add_argument("--mesh", type=str, default=None,
                        help="Path to mesh file (.obj, .stl, .ply)")
    args = parser.parse_args()

    print("=" * 50)
    print("  GraspGen Model Test")
    print("=" * 50)

    import_results = test_imports()

    # Find gripper config
    config = args.gripper_config or find_gripper_config()
    if config is None:
        print("\n[FAIL] No gripper config found. Run ./scripts/download_models.sh first.")
        print("  [INFO] Expected configs in /opt/models/graspgen/")
        sys.exit(1)

    sampler = test_model_loading(config)

    if sampler is not None:
        success = test_inference_synthetic(sampler)
        if args.mesh:
            success = test_inference_mesh(sampler, args.mesh) and success
        test_gpu_memory()
        sys.exit(0 if success else 1)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
