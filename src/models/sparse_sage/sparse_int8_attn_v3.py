"""Block-sparse INT8 attention for FlashVSR on XPU (v3, from /root/xzh/attn_bench/kernels/sage_v3.py).

Same math as sparse_int8_attn.py (bit-identical output), loads rewritten for Intel Triton.

On this Intel Triton build only tl.make_tensor_descriptor loads are lowered to 2D block IO (see probe_2dio.py);
tl.make_block_ptr (v2) is rewritten to plain pointers and ends up as scalar loads.

Same math as the plugin's sparse_sage kernel (INT8 Q.K^T with per-block scales, exp2 softmax, fp16 P.V
accumulated into fp32), so outputs should match it. Changes, each switchable for ablation:
  * Q/K/V loaded through tl.make_tensor_descriptor (2D block IO); K row-major, transposed inside the dot
  * USE_IDX: iterate a compacted list of kept key blocks per query block instead of scanning every block
    and branching on the mask (no data-dependent branch inside the loop)
  * EVEN_N: drop the key-padding mask when kv_len is a multiple of BLOCK_N (descriptors handle the loads)

Non-causal only (FlashVSR never uses causal here). Layout HND: q/k/v [B, H, L, D].
"""

import os

import torch
import triton
import triton.language as tl

BLOCK_M, BLOCK_N = 128, 64  # fixed by FlashVSR's mask granularity


@triton.jit
def _attn_fwd_v3(Q, K, V, Q_scale, K_scale, Mask, KIdx, KCnt, Out,
                 stride_qz, stride_qh, stride_qn,
                 stride_kz, stride_kh, stride_kn,
                 stride_vz, stride_vh, stride_vn,
                 stride_oz, stride_oh, stride_on,
                 stride_mz, stride_mh, stride_mm,
                 stride_iz, stride_ih, stride_im,
                 stride_cz, stride_ch,
                 qo_len, kv_len, H: tl.constexpr, num_kv_groups: tl.constexpr,
                 HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                 USE_IDX: tl.constexpr, EVEN_M: tl.constexpr, EVEN_N: tl.constexpr):
    start_m = tl.program_id(0)
    off_h = tl.program_id(1).to(tl.int64)
    off_z = tl.program_id(2).to(tl.int64)
    kvh = off_h // num_kv_groups
    n_qblk = tl.cdiv(qo_len, BLOCK_M)
    n_kblk = tl.cdiv(kv_len, BLOCK_N)

    qd = tl.make_tensor_descriptor(Q + off_z * stride_qz + off_h * stride_qh, shape=[qo_len, HEAD_DIM],
                                   strides=[stride_qn, 1], block_shape=[BLOCK_M, HEAD_DIM])
    kd = tl.make_tensor_descriptor(K + off_z * stride_kz + kvh * stride_kh, shape=[kv_len, HEAD_DIM],
                                   strides=[stride_kn, 1], block_shape=[BLOCK_N, HEAD_DIM])
    vd = tl.make_tensor_descriptor(V + off_z * stride_vz + kvh * stride_vh, shape=[kv_len, HEAD_DIM],
                                   strides=[stride_vn, 1], block_shape=[BLOCK_N, HEAD_DIM])
    od = tl.make_tensor_descriptor(Out + off_z * stride_oz + off_h * stride_oh, shape=[qo_len, HEAD_DIM],
                                   strides=[stride_on, 1], block_shape=[BLOCK_M, HEAD_DIM])

    q = qd.load([start_m * BLOCK_M, 0])  # descriptors zero-fill out-of-bounds rows
    q_scale = tl.load(Q_scale + (off_z * H + off_h) * n_qblk + start_m)
    k_scale_base = K_scale + (off_z * (H // num_kv_groups) + kvh) * n_kblk

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    if USE_IDX:
        idx_ptr = KIdx + off_z * stride_iz + kvh * stride_ih + start_m * stride_im
        n_iter = tl.load(KCnt + off_z * stride_cz + kvh * stride_ch + start_m)
    else:
        mask_ptr = Mask + off_z * stride_mz + kvh * stride_mh + start_m * stride_mm
        n_iter = n_kblk

    for it in range(0, n_iter):
        if USE_IDX:
            kb = tl.load(idx_ptr + it)
            keep = True
        else:
            kb = it
            keep = tl.load(mask_ptr + it) != 0
        if keep:
            start_n = kb * BLOCK_N
            k = kd.load([start_n, 0])
            v = vd.load([start_n, 0])
            k_scale = tl.load(k_scale_base + kb)
            qk = tl.dot(q, tl.trans(k)).to(tl.float32) * q_scale * k_scale
            if not EVEN_N:
                qk = tl.where((start_n + tl.arange(0, BLOCK_N))[None, :] < kv_len, qk, -float("inf"))
            new_m = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - new_m[:, None]
            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp2(m_i - new_m)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            acc += tl.dot(p.to(tl.float16), v, out_dtype=tl.float16)
            m_i = new_m

    acc = acc / l_i[:, None]
    od.store([start_m * BLOCK_M, 0], acc.to(Out.type.element_ty))  # out-of-bounds rows are dropped


def build_index(mask_id):
    """[B, H, QB, KB] int8 mask -> (idx [B, H, QB, KB] int32 kept key blocks first, ascending; cnt [B, H, QB] int32)."""
    kb = mask_id.shape[-1]
    ar = torch.arange(kb, device=mask_id.device, dtype=torch.int32)
    key = torch.where(mask_id != 0, ar, ar + kb)  # kept blocks sort first, original order preserved
    idx = torch.sort(key, dim=-1).values
    idx = torch.where(idx >= kb, idx - kb, idx).to(torch.int32)
    cnt = (mask_id != 0).sum(-1, dtype=torch.int32)
    return idx.contiguous(), cnt.contiguous()


def supported(q_int8, k_int8, v, is_causal=False, tensor_layout="HND"):
    """Tensor descriptors need 16-byte aligned base pointers and row strides; this kernel is HND / non-causal / D=128."""
    if os.environ.get("FLASHVSR_SAGE_V3", "1") == "0":
        return False
    if is_causal or tensor_layout != "HND" or q_int8.device.type != "xpu" or q_int8.shape[-1] != 128 or v.dtype != torch.float16:
        return False
    for t in (q_int8, k_int8, v):
        if t.stride(-1) != 1 or t.data_ptr() % 16 or any(s * t.element_size() % 16 for s in t.stride()[:-1]):
            return False
    return True


def forward(q_int8, k_int8, mask_id, v, q_scale, k_scale, output_dtype=torch.bfloat16, use_idx=True, index=None,
            num_warps=16, num_stages=2, grf_mode="auto"):
    b, h, qo_len, head_dim = q_int8.shape
    _, h_kv, kv_len, _ = k_int8.shape
    assert v.dtype == torch.float16 and q_int8.stride(-1) == 1 and k_int8.stride(-1) == 1 and v.stride(-1) == 1
    o = torch.empty(q_int8.shape, dtype=output_dtype, device=q_int8.device)
    if use_idx:
        idx, cnt = index if index is not None else build_index(mask_id)
    else:
        idx, cnt = mask_id, mask_id[..., 0]  # unused placeholders with valid strides
    grid = (triton.cdiv(qo_len, BLOCK_M), h, b)
    _attn_fwd_v3[grid](
        q_int8, k_int8, v, q_scale, k_scale, mask_id, idx, cnt, o,
        q_int8.stride(0), q_int8.stride(1), q_int8.stride(2),
        k_int8.stride(0), k_int8.stride(1), k_int8.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        mask_id.stride(0), mask_id.stride(1), mask_id.stride(2),
        idx.stride(0), idx.stride(1), idx.stride(2),
        cnt.stride(0), cnt.stride(1),
        qo_len, kv_len, h, h // h_kv,
        HEAD_DIM=head_dim, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        USE_IDX=use_idx, EVEN_M=qo_len % BLOCK_M == 0, EVEN_N=kv_len % BLOCK_N == 0,
        num_warps=num_warps, num_stages=num_stages, **({"grf_mode": grf_mode} if grf_mode else {}))
    return o

