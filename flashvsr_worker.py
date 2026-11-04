#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Worker process for multi_device.py: owns one device (already pinned by the parent via ZE_AFFINITY_MASK /
# CUDA_VISIBLE_DEVICES, so it is always <type>:0 here), loads a FlashVSR pipeline and upscales tiles on request.

import os
import sys
import time
import traceback
import importlib.util
from multiprocessing.connection import Client


def load_plugin(comfy_root):
    # import the plugin as a package without going through ComfyUI's custom node loader
    plugin_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, comfy_root)
    spec = importlib.util.spec_from_file_location(
        "flashvsr_worker_pkg", os.path.join(plugin_dir, "__init__.py"), submodule_search_locations=[plugin_dir])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pkg
    spec.loader.exec_module(pkg)
    return sys.modules["flashvsr_worker_pkg.nodes"]


def main():
    address, authkey, comfy_root, rank, dev_type = sys.argv[1:6]
    # ComfyUI parses sys.argv on import; the worker has no ComfyUI arguments of its own
    sys.argv = [sys.argv[0]]
    conn = Client(address, family="AF_UNIX", authkey=bytes.fromhex(authkey))
    conn.send(int(rank))
    cfg = conn.recv()

    try:
        import numpy as np
        import torch
        nodes = load_plugin(comfy_root)
        import folder_paths
        for path in cfg["vae_paths"]:
            folder_paths.add_model_folder_path("vae", path)

        device = f"{dev_type}:0"
        if dev_type == "xpu":
            torch.xpu.set_device(device)
        elif dev_type == "cuda":
            torch.cuda.set_device(device)
        nodes.wan_video_dit.USE_BLOCK_ATTN = cfg["attention_mode"] != "sparse_sage_attention"
        dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(cfg["precision"], torch.bfloat16)
        pipe = nodes.init_pipeline(cfg["model"], cfg["mode"], device, dtype, alt_vae=cfg["alt_vae"], model_path=cfg["model_path"])
        # init_pipeline leaves the weights offloaded; release the cached blocks so the device is free for the rest of the workflow
        nodes.clean_vram()
        conn.send(("ready",))
    except Exception:
        conn.send(("error", traceback.format_exc()))
        return

    while True:
        msg = conn.recv()
        if msg[0] == "stop":
            break
        try:
            if msg[0] == "offload":
                pipe.offload_model()
                nodes.clean_vram()
                conn.send(("offloaded",))
                continue
            _, i, frames_path, (x1, y1, x2, y2), p, out_path = msg
            start = time.time()
            frames = torch.from_numpy(np.ascontiguousarray(np.load(frames_path, mmap_mode="r")[:, y1:y2, x1:x2, :]))
            # weights stay on the device between tiles; the parent asks for an offload once the whole video is done
            output = nodes.flashvsr(pipe, frames, p["scale"], p["color_fix"], p["tiled_vae"], False, p["tile_size"],
                                    p["tile_overlap"], p["unload_dit"], p["sparse_ratio"], p["kv_ratio"],
                                    p["local_range"], p["seed"], False)
            np.save(out_path, output.numpy())
            conn.send(("tile", i, out_path, time.time() - start))
        except Exception:
            conn.send(("error", traceback.format_exc()))


if __name__ == "__main__":
    main()
