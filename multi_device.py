#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Tile-parallel FlashVSR across several devices.
# One persistent worker process per device (pinned with ZE_AFFINITY_MASK / CUDA_VISIBLE_DEVICES), each holding its own
# FlashVSR pipeline. The parent pads the frames once into shared memory, hands out tiles to idle workers and blends the
# results in tile order, so the output matches the single-device tiled path.
# Enabled by the "devices" input of FlashVSRInitPipe, e.g. "0,1" or "xpu:0,xpu:1".

import os
import sys
import time
import uuid
import atexit
import shutil
import tempfile
import threading
import subprocess

import numpy as np
import torch
from multiprocessing.connection import Listener, wait

import folder_paths

WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flashvsr_worker.py")
STARTUP_TIMEOUT = 1800

_pools = {}


def parse_devices(spec):
    spec = (spec or "").strip()
    if not spec:
        return []
    devices = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            dev_type, idx = item.split(":", 1)
        else:
            dev_type = "cuda" if torch.cuda.is_available() else "xpu"
            idx = item
        devices.append((dev_type, int(idx)))
    return devices


def _worker_env(dev_type, idx):
    env = os.environ.copy()
    mask_var = "ZE_AFFINITY_MASK" if dev_type == "xpu" else "CUDA_VISIBLE_DEVICES"
    # Respect an existing mask: the device index is relative to what this process can see
    parent_mask = [m.strip() for m in env.get(mask_var, "").split(",") if m.strip()]
    env[mask_var] = parent_mask[idx] if parent_mask else str(idx)
    return env


class FlashVSRDevicePool:
    def __init__(self, devices, cfg):
        self.devices = devices
        self.cfg = cfg
        self.procs = []
        self.conns = [None] * len(devices)
        self.ready = [False] * len(devices)
        self.closed = False
        self.tmp_dir = tempfile.mkdtemp(prefix="flashvsr_", dir="/dev/shm" if os.path.isdir("/dev/shm") else None)
        self.authkey = os.urandom(16)
        self.listener = Listener(os.path.join(self.tmp_dir, "sock"), family="AF_UNIX", authkey=self.authkey)
        comfy_root = os.path.dirname(os.path.abspath(folder_paths.__file__))
        for rank, (dev_type, idx) in enumerate(devices):
            cmd = [sys.executable, WORKER_SCRIPT, self.listener.address, self.authkey.hex(), comfy_root, str(rank), dev_type]
            self.procs.append(subprocess.Popen(cmd, env=_worker_env(dev_type, idx)))
        self.accept_thread = threading.Thread(target=self._accept, daemon=True)
        self.accept_thread.start()

    def _accept(self):
        for _ in self.devices:
            try:
                conn = self.listener.accept()
            except Exception:
                return
            rank = conn.recv()
            conn.send(self.cfg)
            self.conns[rank] = conn

    def _check_alive(self):
        for rank, proc in enumerate(self.procs):
            code = proc.poll()
            if code is not None:
                raise RuntimeError(f"[FlashVSR] worker {rank} ({self.device_name(rank)}) exited with code {code}")

    def device_name(self, rank):
        dev_type, idx = self.devices[rank]
        return f"{dev_type}:{idx}"

    def _recv(self, rank):
        msg = self.conns[rank].recv()
        if msg[0] == "error":
            raise RuntimeError(f"[FlashVSR] worker {rank} ({self.device_name(rank)}) failed:\n{msg[-1]}")
        return msg

    def wait_ready(self):
        deadline = time.time() + STARTUP_TIMEOUT
        while not all(self.ready):
            self._check_alive()
            if time.time() > deadline:
                raise RuntimeError("[FlashVSR] timed out waiting for workers to load the model")
            conns = [c for r, c in enumerate(self.conns) if c is not None and not self.ready[r]]
            if not conns:
                time.sleep(0.2)
                continue
            for conn in wait(conns, timeout=1.0):
                rank = self.conns.index(conn)
                self._recv(rank)
                self.ready[rank] = True

    def run_tiles(self, frames_path, jobs, params, on_result):
        """jobs: list of (x1, y1, x2, y2). on_result(i, tile) is called in tile order."""
        self.wait_ready()
        pending = list(range(len(jobs)))
        busy = {}
        done = {}
        next_blend = 0

        def dispatch(rank):
            i = pending.pop(0)
            out_path = os.path.join(self.tmp_dir, f"tile_{uuid.uuid4().hex}.npy")
            self.conns[rank].send(("tile", i, frames_path, jobs[i], params, out_path))
            busy[rank] = i

        for rank in range(len(self.devices)):
            if pending:
                dispatch(rank)

        while busy:
            self._check_alive()
            for conn in wait([self.conns[r] for r in busy], timeout=1.0):
                rank = self.conns.index(conn)
                _, i, out_path, elapsed = self._recv(rank)
                del busy[rank]
                print(f"[FlashVSR] tile {i + 1}/{len(jobs)} done on {self.device_name(rank)} in {elapsed:.1f}s")
                done[i] = out_path
                if pending:
                    dispatch(rank)
            while next_blend in done:
                path = done.pop(next_blend)
                on_result(next_blend, torch.from_numpy(np.load(path)))
                os.remove(path)
                next_blend += 1

    def offload(self):
        for conn in self.conns:
            conn.send(("offload",))
        for rank in range(len(self.devices)):
            self._recv(rank)

    def close(self):
        if self.closed:
            return
        self.closed = True
        for conn in self.conns:
            try:
                if conn is not None:
                    conn.send(("stop",))
            except Exception:
                pass
        for proc in self.procs:
            try:
                proc.wait(timeout=30)
            except Exception:
                proc.kill()
        try:
            self.listener.close()
        except Exception:
            pass
        shutil.rmtree(self.tmp_dir, ignore_errors=True)


def get_pool(devices, cfg):
    key = (tuple(devices), tuple(sorted(cfg.items())))
    pool = _pools.get(key)
    if pool is not None and (pool.closed or any(p.poll() is not None for p in pool.procs)):
        pool.close()
        pool = None
    if pool is None:
        # only one configuration is kept alive, so a changed workflow does not leave stale workers holding VRAM
        for other in list(_pools):
            _pools.pop(other).close()
        pool = FlashVSRDevicePool(devices, cfg)
        _pools[key] = pool
    return pool


def close_all():
    for key in list(_pools):
        _pools.pop(key).close()


atexit.register(close_all)


class MultiDevicePipe:
    """Returned by FlashVSRInitPipe instead of a pipeline when "devices" lists several devices."""

    def __init__(self, devices, cfg):
        self.devices = devices
        self.cfg = cfg
        # start loading right away so model loading overlaps with the rest of the workflow
        get_pool(devices, cfg)

    @property
    def pool(self):
        return get_pool(self.devices, self.cfg)


def flashvsr_multi(mpipe, frames, scale, color_fix, tiled_vae, tiled_dit, tile_size, tile_overlap, unload_dit,
                   sparse_ratio, kv_ratio, local_range, seed, force_offload):
    from .nodes import free_comfy_vram, next_8n5, calculate_tile_coords, create_feather_mask, log

    # the workers live in other processes, but the parent's ComfyUI models may still sit on one of their devices
    free_comfy_vram()
    pool = mpipe.pool

    add = next_8n5(frames.shape[0]) - frames.shape[0]
    _frames = torch.cat([frames, frames[-1:, :, :, :].repeat(add, 1, 1, 1)], dim=0)
    N, H, W, C = _frames.shape

    frames_path = os.path.join(pool.tmp_dir, f"frames_{uuid.uuid4().hex}.npy")
    np.save(frames_path, _frames.float().cpu().numpy())

    params = dict(scale=scale, color_fix=color_fix, tiled_vae=tiled_vae, tile_size=tile_size,
                  tile_overlap=tile_overlap, unload_dit=unload_dit, sparse_ratio=sparse_ratio,
                  kv_ratio=kv_ratio, local_range=local_range, seed=seed)

    try:
        if tiled_dit:
            tile_coords = calculate_tile_coords(H, W, tile_size, tile_overlap)
            log(f"[FlashVSR] Processing {len(tile_coords)} tiles on {len(pool.devices)} devices: "
                f"{', '.join(pool.device_name(r) for r in range(len(pool.devices)))}", message_type='info')
            final_output_canvas = torch.zeros((N, H * scale, W * scale, C), dtype=torch.float16, device="cpu")
            weight_sum_canvas = torch.zeros_like(final_output_canvas)

            def blend(i, processed_tile_cpu):
                x1, y1, x2, y2 = tile_coords[i]
                mask_nchw = create_feather_mask(
                    (processed_tile_cpu.shape[1], processed_tile_cpu.shape[2]),
                    tile_overlap * scale
                ).to("cpu")
                mask_nhwc = mask_nchw.permute(0, 2, 3, 1)
                out_x1, out_y1 = x1 * scale, y1 * scale
                out_x2, out_y2 = out_x1 + processed_tile_cpu.shape[2], out_y1 + processed_tile_cpu.shape[1]
                final_output_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += processed_tile_cpu * mask_nhwc
                weight_sum_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += mask_nhwc

            pool.run_tiles(frames_path, tile_coords, params, blend)
            weight_sum_canvas[weight_sum_canvas == 0] = 1.0
            final_output = final_output_canvas / weight_sum_canvas
        else:
            # a single tile covering the whole frame: nothing to parallelize, run it on the first free worker
            result = {}
            pool.run_tiles(frames_path, [(0, 0, W, H)], params, lambda i, t: result.setdefault(i, t))
            final_output = result[0]
    finally:
        if os.path.exists(frames_path):
            os.remove(frames_path)

    if force_offload:
        pool.offload()

    log("[FlashVSR] Done.", message_type='info')
    if frames.shape[0] == 1:
        return torch.median(final_output, dim=0).values.unsqueeze(0).float()
    return final_output[:frames.shape[0], :, :, :]
