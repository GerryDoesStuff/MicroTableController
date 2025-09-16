#!/usr/bin/env python3
"""
Microstage EDF (GPU-enabled):
- CPU baseline identical to your current version.
- Optional CUDA acceleration via OpenCV's cv2.cuda_* when available.

RESTORED DEFAULTS (as requested previously):
 - color-mode: lab_l_only
 - linear-fusion: OFF by default (enable with --linear-fusion)
 - sat-weight: OFF by default (enable with --sat-weight)
 - illumination: none by default (enable with --illum stack_median or --illum per_slice)
 - clahe-l: OFF by default (enable with --clahe-l)

GPU:
 - --gpu enables GPU path (requires CUDA-enabled OpenCV).
 - Falls back to CPU automatically if CUDA features are unavailable.

Outputs named "<folder>-deep.*".
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import cv2
import tifffile as tiff

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# --------------------- GPU helpers ---------------------
def has_cuda() -> bool:
    return (hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0)

class CudaKernels:
    """Lazily creates CUDA filters/operators we need."""
    def __init__(self):
        self.stream = cv2.cuda.Stream()
        self.gauss_cache = {}
        self.sobel_cache = {}
        self.laplace_cache = {}
        self.clahe = None

    def gaussian(self, ksize: int, sigma: float):
        key = (ksize, sigma)
        if key not in self.gauss_cache:
            # cv2.CV_32F filter output type decided by src depth; we pass GpuMat with float32 or uint8
            self.gauss_cache[key] = cv2.cuda.createGaussianFilter(
                srcType=cv2.CV_32F, dstType=cv2.CV_32F, ksize=(ksize, ksize), sigma1=sigma, sigma2=sigma
            )
        return self.gauss_cache[key]

    def sobel(self, dx: int, dy: int, ksize: int):
        key = (dx, dy, ksize)
        if key not in self.sobel_cache:
            self.sobel_cache[key] = cv2.cuda.createSobelFilter(
                srcType=cv2.CV_32F, dstType=cv2.CV_32F, dx=dx, dy=dy, ksize=ksize
            )
        return self.sobel_cache[key]

    def laplacian(self, ksize: int):
        key = ksize
        if key not in self.laplace_cache:
            self.laplace_cache[key] = cv2.cuda.createLaplacianFilter(
                srcType=cv2.CV_32F, dstType=cv2.CV_32F, ksize=ksize, scale=1.0
            )
        return self.laplace_cache[key]

    def get_clahe(self, clip_limit: float, tiles: int):
        # CUDA CLAHE expects 8u input
        if self.clahe is None:
            self.clahe = cv2.cuda.createCLAHE(clipLimit=clip_limit, tileGridSize=(tiles, tiles))
        else:
            # recreate if params changed
            pass
        return self.clahe

# --------------------- File discovery & sorting ---------------------
def find_stack_images(folder: Path, pattern: Optional[str]) -> List[Path]:
    if pattern:
        candidates = list(folder.glob(pattern))
    else:
        candidates = [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS]
    def key(p: Path):
        m = re.search(r"(\d{4,})", p.stem)
        primary = int(m.group(1)) if m else float("inf")
        return (primary, p.name.lower())
    return sorted([p for p in candidates if p.is_file()], key=key)

# --------------------------- I/O utilities --------------------------
def read_image(path: Path) -> np.ndarray:
    """Read with OpenCV; keep BGR order. Drop alpha if present."""
    data = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if data is None:
        raise RuntimeError(f"Failed to read {path}")
    if data.ndim == 2:
        return data
    if data.shape[2] == 4:
        data = cv2.cvtColor(data, cv2.COLOR_BGRA2BGR)
    return data  # BGR

def ensure_same_channels(images: List[np.ndarray]) -> bool:
    chs = [(1 if im.ndim == 2 else im.shape[2]) for im in images]
    return len(set(chs)) == 1

def harmonize_sizes(images: List[np.ndarray]) -> List[np.ndarray]:
    h_min = min(im.shape[0] for im in images)
    w_min = min(im.shape[1] for im in images)
    return [cv2.resize(im, (w_min, h_min), interpolation=cv2.INTER_AREA) for im in images]

def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32)

# ------------------------- sRGB <-> linear helpers ------------------
def srgb_to_linear(img_8u_bgr: np.ndarray) -> np.ndarray:
    """Input: uint8 BGR; Output: float32 linear RGB in [0,1] (3 channels)."""
    if img_8u_bgr.ndim == 2:
        x = img_8u_bgr.astype(np.float32) / 255.0
        x = np.stack([x, x, x], axis=-1)
        bgr = x[..., ::-1]
    else:
        bgr = img_8u_bgr.astype(np.float32) / 255.0
    rgb = bgr[..., ::-1]  # BGR->RGB
    a = 0.055
    thresh = 0.04045
    lin = np.where(rgb <= thresh, rgb / 12.92, ((rgb + a) / (1 + a)) ** 2.4)
    return lin.astype(np.float32)

def linear_to_srgb(rgb_lin: np.ndarray) -> np.ndarray:
    """Input: float32 linear RGB [0,1]; Output: uint8 BGR in sRGB encoding."""
    a = 0.055
    thresh = 0.0031308
    srgb = np.where(rgb_lin <= thresh, 12.92 * rgb_lin, (1 + a) * (rgb_lin ** (1/2.4)) - a)
    srgb = np.clip(srgb, 0.0, 1.0)
    bgr8 = (srgb[..., ::-1] * 255.0 + 0.5).astype(np.uint8)  # RGB->BGR
    return bgr8

# ------------------------- Illumination correction ------------------
def estimate_flatfield_stack_median(stack: List[np.ndarray], blur_sigma: float = 50.0) -> np.ndarray:
    """Per-channel flat-field as the median across the stack, then heavy Gaussian blur."""
    arr = np.stack(stack, axis=0).astype(np.float32)
    if arr.ndim == 4:
        med = np.median(arr, axis=0)
        for c in range(med.shape[2]):
            med[..., c] = cv2.GaussianBlur(med[..., c], (0, 0), blur_sigma)
    else:
        med = np.median(arr, axis=0)
        med = cv2.GaussianBlur(med, (0, 0), blur_sigma)
    return np.clip(med, 1e-6, None)

def estimate_flatfield_per_slice(img: np.ndarray, blur_sigma: float = 50.0) -> np.ndarray:
    """Per-slice flat-field via heavy Gaussian blur (low-pass illumination)."""
    if img.ndim == 3:
        out = np.empty_like(img, dtype=np.float32)
        for c in range(img.shape[2]):
            out[..., c] = cv2.GaussianBlur(img[..., c].astype(np.float32), (0, 0), blur_sigma)
    else:
        out = cv2.GaussianBlur(img.astype(np.float32), (0, 0), blur_sigma)
    return np.clip(out, 1e-6, None)

def apply_flatfield(img: np.ndarray, flat: np.ndarray) -> np.ndarray:
    """Divide by flat-field and rescale by its mean (per-channel aware)."""
    imgf = img.astype(np.float32)
    if imgf.ndim == 2 and flat.ndim == 3:
        flat = cv2.cvtColor(flat.astype(np.float32), cv2.COLOR_BGR2GRAY)
    if imgf.ndim == 3 and flat.ndim == 2:
        flat = np.repeat(flat[..., None], imgf.shape[2], axis=2)
    mean_flat = np.mean(flat, axis=(0,1), keepdims=True)
    corrected = imgf * (mean_flat / flat)
    return np.clip(corrected, 0, 255).astype(img.dtype)

# --------------------------- Focus measures -------------------------
def fm_tenengrad(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
    return gx*gx + gy*gy

def fm_variance(gray: np.ndarray, win: int = 7) -> np.ndarray:
    mu = cv2.GaussianBlur(gray, (0, 0), win/6.0)
    mu2 = cv2.GaussianBlur(gray*gray, (0, 0), win/6.0)
    return np.maximum(mu2 - mu*mu, 0)

def fm_laplacian(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=ksize)
    return lap*lap

def fm_brenner(gray: np.ndarray) -> np.ndarray:
    dx = gray[:, 2:] - gray[:, :-2]
    dy = gray[2:, :] - gray[:-2, :]
    m = np.zeros_like(gray)
    m[:, 1:-1] += 0.5*(dx[:, :-1]**2 + dx[:, 1:]**2)
    m[1:-1, :] += 0.5*(dy[:-1, :]**2 + dy[1:, :]**2)
    return m

def fm_vollath_f4(gray: np.ndarray) -> np.ndarray:
    s1 = gray[1:, :] * gray[:-1, :]
    s2 = gray[2:, :] * gray[:-2, :]
    m = np.zeros_like(gray)
    m[1:, :] += s1
    m[2:, :] -= s2
    return np.maximum(m, 0)

FOCUS_FUNCS = {
    "tenengrad":  fm_tenengrad,
    "variance":   fm_variance,
    "laplacian":  fm_laplacian,
    "brenner":    fm_brenner,
    "vollath_f4": fm_vollath_f4,
}

# -------------------------- Multiscale helpers ----------------------
def normalize_weights(weight_maps: List[np.ndarray], blur_sigma: float = 1.0, eps: float = 1e-12) -> List[np.ndarray]:
    W = np.stack(weight_maps, axis=0)
    W = np.maximum(W, 0)
    for i in range(W.shape[0]):
        W[i] = cv2.GaussianBlur(W[i], (0, 0), blur_sigma)
    denom = np.sum(W, axis=0, keepdims=True) + eps
    Wn = W / denom
    return [Wn[i] for i in range(Wn.shape[0])]

def build_gaussian_pyr(img: np.ndarray, levels: int) -> List[np.ndarray]:
    G = [img]
    for _ in range(1, levels):
        G.append(cv2.pyrDown(G[-1]))
    return G

def build_laplacian_pyr(img: np.ndarray, levels: int) -> List[np.ndarray]:
    G = build_gaussian_pyr(img, levels)
    L = []
    for i in range(levels-1):
        up = cv2.pyrUp(G[i+1], dstsize=(G[i].shape[1], G[i].shape[0]))
        L.append(G[i].astype(np.float32) - up.astype(np.float32))
    L.append(G[-1].astype(np.float32))
    return L

def reconstruct_from_laplacian(L: List[np.ndarray]) -> np.ndarray:
    img = L[-1]
    for i in range(len(L)-2, -1, -1):
        up = cv2.pyrUp(img, dstsize=(L[i].shape[1], L[i].shape[0]))
        img = up + L[i]
    return img

# ---------------------- CUDA variants of key ops --------------------
def cuda_gaussian_blur_float(gm: cv2.cuda_GpuMat, sigma: float, kernels: CudaKernels) -> cv2.cuda_GpuMat:
    # choose ksize based on sigma (OpenCV allows ksize=0 and sigma>0 in CPU; here we pick an odd size)
    ksize = max(3, int(2 * round(3*sigma) + 1))
    filt = kernels.gaussian(ksize, sigma)
    dst = cv2.cuda_GpuMat()
    filt.apply(gm, dst, stream=kernels.stream)
    return dst

def cuda_tenengrad(gray8u_gm: cv2.cuda_GpuMat, ksize: int, kernels: CudaKernels) -> cv2.cuda_GpuMat:
    # convert to float
    gray32f = cv2.cuda_GpuMat()
    gray8u_gm.convertTo(gray32f, cv2.CV_32F, stream=kernels.stream)
    gx = cv2.cuda_GpuMat(); gy = cv2.cuda_GpuMat()
    kernels.sobel(1,0,ksize).apply(gray32f, gx, stream=kernels.stream)
    kernels.sobel(0,1,ksize).apply(gray32f, gy, stream=kernels.stream)
    # fm = gx*gx + gy*gy
    gx2 = cv2.cuda.multiply(gx, gx, stream=kernels.stream)
    gy2 = cv2.cuda.multiply(gy, gy, stream=kernels.stream)
    fm  = cv2.cuda.add(gx2, gy2, stream=kernels.stream)
    return fm  # CV_32F

def cuda_gray(image_bgr_gm: cv2.cuda_GpuMat, kernels: CudaKernels) -> cv2.cuda_GpuMat:
    gray = cv2.cuda.cvtColor(image_bgr_gm, cv2.COLOR_BGR2GRAY, stream=kernels.stream)
    return gray

def cuda_build_gaussian_pyr(gm: cv2.cuda_GpuMat, levels: int, kernels: CudaKernels) -> List[cv2.cuda_GpuMat]:
    G = [gm]
    for _ in range(1, levels):
        dst = cv2.cuda_GpuMat()
        cv2.cuda.pyrDown(G[-1], dst, stream=kernels.stream)
        G.append(dst)
    return G

def cuda_build_laplacian_pyr(gm: cv2.cuda_GpuMat, levels: int, kernels: CudaKernels) -> List[cv2.cuda_GpuMat]:
    G = cuda_build_gaussian_pyr(gm, levels, kernels)
    L = []
    for i in range(levels-1):
        up = cv2.cuda_GpuMat()
        cv2.cuda.pyrUp(G[i+1], up, stream=kernels.stream)
        # ensure same size
        up = cv2.cuda.resize(up, (G[i].cols(), G[i].rows()), stream=kernels.stream)
        diff = cv2.cuda.subtract(G[i], up, stream=kernels.stream)
        L.append(diff)
    L.append(G[-1])
    return L

def cuda_reconstruct_from_laplacian(L: List[cv2.cuda_GpuMat], kernels: CudaKernels) -> cv2.cuda_GpuMat:
    img = L[-1]
    for i in range(len(L)-2, -1, -1):
        up = cv2.cuda_GpuMat()
        cv2.cuda.pyrUp(img, up, stream=kernels.stream)
        up = cv2.cuda.resize(up, (L[i].cols(), L[i].rows()), stream=kernels.stream)
        img = cv2.cuda.add(up, L[i], stream=kernels.stream)
    return img

# -------------------------- Exposure-fusion weights -----------------
def saturation_map_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """HSV saturation (0..1) used for optional exposure-fusion-style weighting."""
    if img_bgr.ndim == 2:
        return np.zeros_like(img_bgr, dtype=np.float32)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    S = hsv[..., 1].astype(np.float32) / 255.0
    return S

# -------------------------- Fusion wrappers (CPU) -------------------
def compute_focus_and_weights_cpu(images: List[np.ndarray],
                                  focus: str,
                                  sobel_ksize: int,
                                  use_sat_weight: bool,
                                  sat_gamma: float) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray]]:
    grays = [to_gray(im) for im in images]
    if focus == "tenengrad":
        fmaps = [FOCUS_FUNCS[focus](g, ksize=sobel_ksize) for g in grays]
    elif focus in ("laplacian",):
        fmaps = [FOCUS_FUNCS[focus](g, ksize=3) for g in grays]
    else:
        fmaps = [FOCUS_FUNCS[focus](g) for g in grays]
    depth_idx = np.argmax(np.stack(fmaps, axis=0), axis=0).astype(np.uint16)

    if use_sat_weight and images[0].ndim == 3:
        sats = [saturation_map_bgr(im) for im in images]
        sats = [cv2.GaussianBlur(s, (0, 0), 0.8) for s in sats]
        fmaps = [fm * (np.power(s + 1e-6, sat_gamma)) for fm, s in zip(fmaps, sats)]

    weights = normalize_weights(fmaps, blur_sigma=1.0)
    return fmaps, depth_idx, weights

def fuse_luminance_pyramid_cpu(maps: List[np.ndarray], weights: List[np.ndarray], levels: int) -> np.ndarray:
    L_pyrs = [build_laplacian_pyr(m.astype(np.float32), levels) for m in maps]
    W_pyrs = [build_gaussian_pyr(w, levels) for w in weights]
    fused_pyr = []
    for lvl in range(levels):
        acc = np.zeros_like(L_pyrs[0][lvl], dtype=np.float32)
        for s in range(len(maps)):
            acc += L_pyrs[s][lvl] * W_pyrs[s][lvl]
        fused_pyr.append(acc)
    fused = reconstruct_from_laplacian(fused_pyr)
    return np.clip(fused, 0, 255).astype(np.uint8)

# -------------------------- Fusion wrappers (CUDA) ------------------
def compute_focus_and_weights_cuda(images_bgr: List[np.ndarray],
                                   focus: str,
                                   sobel_ksize: int,
                                   use_sat_weight: bool,
                                   sat_gamma: float,
                                   kernels: CudaKernels) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray]]:
    """
    Returns CPU arrays: fmaps (list of float32 HxW), depth_idx (uint16), weights (list of float32 HxW).
    We keep weight logic on CPU to simplify, but compute focus maps on GPU.
    """
    fmaps_gpu = []
    for im in images_bgr:
        gm = cv2.cuda_GpuMat()
        gm.upload(im, stream=kernels.stream)
        gray_gm = cuda_gray(gm, kernels)
        # CUDA CLAHE for focus stability? (off by default)
        # Build focus (Tenengrad default)
        if focus == "tenengrad":
            fm_gm = cuda_tenengrad(gray_gm, sobel_ksize, kernels)
        elif focus == "laplacian":
            # Laplacian focus (CV_32F)
            gray32 = cv2.cuda_GpuMat(); gray_gm.convertTo(gray32, cv2.CV_32F, stream=kernels.stream)
            fm_gm = kernels.laplacian(3).apply(gray32, stream=kernels.stream)
            fm_gm = cv2.cuda.multiply(fm_gm, fm_gm, stream=kernels.stream)
        else:
            # For other measures, fall back to CPU for focus map
            gray_cpu = gray_gm.download(kernels.stream)
            fm_cpu = FOCUS_FUNCS[focus](gray_cpu.astype(np.float32))
            fmaps_gpu.append(fm_cpu)
            continue
        fmaps_gpu.append(fm_gm.download(kernels.stream))

    fmaps = [fm.astype(np.float32) for fm in fmaps_gpu]
    depth_idx = np.argmax(np.stack(fmaps, axis=0), axis=0).astype(np.uint16)

    if use_sat_weight and images_bgr[0].ndim == 3:
        sats = [saturation_map_bgr(im) for im in images_bgr]
        sats = [cv2.GaussianBlur(s, (0, 0), 0.8) for s in sats]
        fmaps = [fm * (np.power(s + 1e-6, sat_gamma)) for fm, s in zip(fmaps, sats)]

    weights = normalize_weights(fmaps, blur_sigma=1.0)
    return fmaps, depth_idx, weights

def fuse_luminance_pyramid_cuda(maps: List[np.ndarray], weights: List[np.ndarray], levels: int, kernels: CudaKernels) -> np.ndarray:
    """
    Fuse luminance maps using CUDA pyramids. Inputs are CPU float32 HxW; we upload, pyramid, and fuse on GPU.
    """
    # Upload maps
    L_pyrs = []
    for m in maps:
        gm = cv2.cuda_GpuMat()
        gm.upload(m.astype(np.float32))
        L_pyrs.append(cuda_build_laplacian_pyr(gm, levels, kernels))
    W_pyrs = []
    for w in weights:
        gm = cv2.cuda_GpuMat()
        gm.upload(w.astype(np.float32))
        W_pyrs.append(cuda_build_gaussian_pyr(gm, levels, kernels))

    fused_pyr = []
    for lvl in range(levels):
        # acc = sum(L_pyrs[s][lvl] * W_pyrs[s][lvl])
        acc = cv2.cuda_GpuMat()
        acc.create(L_pyrs[0][lvl].size(), cv2.CV_32F)
        acc.setTo(0)
        for s in range(len(maps)):
            prod = cv2.cuda.multiply(L_pyrs[s][lvl], W_pyrs[s][lvl], stream=kernels.stream)
            acc = cv2.cuda.add(acc, prod, stream=kernels.stream)
        fused_pyr.append(acc)

    fused_gm = cuda_reconstruct_from_laplacian(fused_pyr, kernels)
    fused = fused_gm.download(kernels.stream)
    fused = np.clip(fused, 0, 255).astype(np.uint8)
    return fused

# -------------------------- Color-preserving fusion -----------------
def fuse_lab_l_only_cpu(images: List[np.ndarray],
                        weights: List[np.ndarray],
                        depth_idx: np.ndarray,
                        levels: int,
                        linear_fusion: bool,
                        clahe_l: bool,
                        clahe_clip: float,
                        clahe_tiles: int) -> np.ndarray:
    H, W = images[0].shape[:2]

    if linear_fusion:
        Y_list = []
        for im in images:
            rgb_lin = srgb_to_linear(im)
            Y = 0.2126*rgb_lin[...,0] + 0.7152*rgb_lin[...,1] + 0.0722*rgb_lin[...,2]
            Y_list.append((np.clip(Y,0,1)*255.0).astype(np.float32))
        fused_L = fuse_luminance_pyramid_cpu(Y_list, weights, levels).astype(np.float32)/255.0
        if clahe_l:
            L_8u = (np.clip(fused_L,0,1)*255.0 + 0.5).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tiles, clahe_tiles))
            L_8u = clahe.apply(L_8u)
            fused_L = L_8u.astype(np.float32)/255.0
        labs = [cv2.cvtColor(im, cv2.COLOR_BGR2Lab) for im in images]
        a_stack = np.stack([lab[...,1] for lab in labs], axis=0)
        b_stack = np.stack([lab[...,2] for lab in labs], axis=0)
        rows = np.arange(H)[:, None]; cols = np.arange(W)[None, :]
        a_sel = a_stack[depth_idx, rows, cols]; b_sel = b_stack[depth_idx, rows, cols]
        bgr8 = linear_to_srgb(np.stack([fused_L, fused_L, fused_L], axis=-1))
        lab_gray = cv2.cvtColor(bgr8, cv2.COLOR_BGR2Lab)
        L_chan = lab_gray[...,0]
        fused_lab = np.dstack([L_chan, a_sel, b_sel]).astype(np.uint8)
        bgr = cv2.cvtColor(fused_lab, cv2.COLOR_Lab2BGR)
        return bgr
    else:
        labs = [cv2.cvtColor(im, cv2.COLOR_BGR2Lab).astype(np.float32) for im in images]
        Ls = [lab[...,0] for lab in labs]
        fused_L = fuse_luminance_pyramid_cpu(Ls, weights, levels)
        if clahe_l:
            clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tiles, clahe_tiles))
            fused_L = clahe.apply(fused_L)
        a_stack = np.stack([lab[...,1] for lab in labs], axis=0).astype(np.uint8)
        b_stack = np.stack([lab[...,2] for lab in labs], axis=0).astype(np.uint8)
        rows = np.arange(H)[:, None]; cols = np.arange(W)[None, :]
        a_sel = a_stack[depth_idx, rows, cols]; b_sel = b_stack[depth_idx, rows, cols]
        fused_lab = np.dstack([fused_L, a_sel, b_sel]).astype(np.uint8)
        bgr = cv2.cvtColor(fused_lab, cv2.COLOR_Lab2BGR)
        return bgr

def fuse_lab_l_only_cuda(images: List[np.ndarray],
                         weights: List[np.ndarray],
                         depth_idx: np.ndarray,
                         levels: int,
                         linear_fusion: bool,
                         clahe_l: bool,
                         clahe_clip: float,
                         clahe_tiles: int,
                         kernels: CudaKernels) -> np.ndarray:
    """CUDA luminance fusion, CPU Lab chroma pick (simple & effective)."""
    H, W = images[0].shape[:2]

    if linear_fusion:
        Y_list = []
        for im in images:
            rgb_lin = srgb_to_linear(im)
            Y = 0.2126*rgb_lin[...,0] + 0.7152*rgb_lin[...,1] + 0.0722*rgb_lin[...,2]
            Y_list.append((np.clip(Y,0,1)*255.0).astype(np.float32))
        fused_L = fuse_luminance_pyramid_cuda(Y_list, weights, levels, kernels).astype(np.float32)/255.0
        if clahe_l:
            L_8u = (np.clip(fused_L,0,1)*255.0 + 0.5).astype(np.uint8)
            # CUDA CLAHE for speed
            gm = cv2.cuda_GpuMat(); gm.upload(L_8u)
            clahe = kernels.get_clahe(clip_limit=clahe_clip, tiles=clahe_tiles)
            L_8u = clahe.apply(gm).download()
            fused_L = L_8u.astype(np.float32)/255.0
        labs = [cv2.cvtColor(im, cv2.COLOR_BGR2Lab) for im in images]
        a_stack = np.stack([lab[...,1] for lab in labs], axis=0)
        b_stack = np.stack([lab[...,2] for lab in labs], axis=0)
        rows = np.arange(H)[:, None]; cols = np.arange(W)[None, :]
        a_sel = a_stack[depth_idx, rows, cols]; b_sel = b_stack[depth_idx, rows, cols]
        bgr8 = linear_to_srgb(np.stack([fused_L, fused_L, fused_L], axis=-1))
        lab_gray = cv2.cvtColor(bgr8, cv2.COLOR_BGR2Lab)
        L_chan = lab_gray[...,0]
        fused_lab = np.dstack([L_chan, a_sel, b_sel]).astype(np.uint8)
        bgr = cv2.cvtColor(fused_lab, cv2.COLOR_Lab2BGR)
        return bgr
    else:
        labs = [cv2.cvtColor(im, cv2.COLOR_BGR2Lab).astype(np.float32) for im in images]
        Ls = [lab[...,0] for lab in labs]
        fused_L = fuse_luminance_pyramid_cuda(Ls, weights, levels, kernels)
        if clahe_l:
            gm = cv2.cuda_GpuMat(); gm.upload(fused_L)
            clahe = kernels.get_clahe(clip_limit=clahe_clip, tiles=clahe_tiles)
            fused_L = clahe.apply(gm).download()
        a_stack = np.stack([lab[...,1] for lab in labs], axis=0).astype(np.uint8)
        b_stack = np.stack([lab[...,2] for lab in labs], axis=0).astype(np.uint8)
        rows = np.arange(H)[:, None]; cols = np.arange(W)[None, :]
        a_sel = a_stack[depth_idx, rows, cols]; b_sel = b_stack[depth_idx, rows, cols]
        fused_lab = np.dstack([fused_L, a_sel, b_sel]).astype(np.uint8)
        bgr = cv2.cvtColor(fused_lab, cv2.COLOR_Lab2BGR)
        return bgr

# ------------------------------ Glue ------------------------------
def fuse_stack(images: List[np.ndarray],
               color_mode: str,
               focus: str,
               levels: int,
               sobel_ksize: int,
               linear_fusion: bool,
               use_sat_weight: bool,
               sat_gamma: float,
               clahe_l: bool,
               clahe_clip: float,
               clahe_tiles: int,
               use_gpu: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (fused_img uint8 BGR, depth_index uint16)
    """
    if images[0].ndim == 2:  # grayscale stack
        if use_gpu and has_cuda():
            kernels = CudaKernels()
            fmaps, depth_idx, weights = compute_focus_and_weights_cuda(images, focus, sobel_ksize, use_sat_weight, sat_gamma, kernels)
            fused = fuse_luminance_pyramid_cuda([im.astype(np.float32) for im in images], weights, levels, kernels)
        else:
            fmaps, depth_idx, weights = compute_focus_and_weights_cpu(images, focus, sobel_ksize, use_sat_weight, sat_gamma)
            fused = fuse_luminance_pyramid_cpu([im.astype(np.float32) for im in images], weights, levels)
        return fused, depth_idx

    if color_mode != "lab_l_only":
        raise ValueError("Only 'lab_l_only' supported in this build.")

    if use_gpu and has_cuda():
        kernels = CudaKernels()
        fmaps, depth_idx, weights = compute_focus_and_weights_cuda(images, focus, sobel_ksize, use_sat_weight, sat_gamma, kernels)
        fused = fuse_lab_l_only_cuda(
            images, weights, depth_idx, levels,
            linear_fusion, clahe_l, clahe_clip, clahe_tiles, kernels
        )
    else:
        fmaps, depth_idx, weights = compute_focus_and_weights_cpu(images, focus, sobel_ksize, use_sat_weight, sat_gamma)
        fused = fuse_lab_l_only_cpu(
            images, weights, depth_idx, levels,
            linear_fusion, clahe_l, clahe_clip, clahe_tiles
        )
    return fused, depth_idx

# ------------------------------ Main --------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="EDF with optional CUDA acceleration. CPU baseline preserved; GPU via cv2.cuda if available."
    )
    ap.add_argument("--pattern", type=str, default=None, help="Glob filter, e.g. 'stack_*.tif'. If omitted, all common image types.")
    ap.add_argument("--focus", type=str, default="tenengrad",
                    choices=list(FOCUS_FUNCS.keys()), help="Focus measure.")
    ap.add_argument("--levels", type=int, default=5, help="Pyramid levels (3–6 typical).")
    ap.add_argument("--sobel-ksize", type=int, default=3, choices=[3,5,7], help="Sobel kernel (Tenengrad).")

    # Illumination correction (DEFAULT: none)
    ap.add_argument("--illum", type=str, default="none",
                    choices=["none","per_slice","stack_median"],
                    help="Illumination correction mode (per-channel).")
    ap.add_argument("--illum-sigma", type=float, default=50.0, help="Gaussian sigma for flat-field estimation.")

    # Color & linear fusion options
    ap.add_argument("--color-mode", type=str, default="lab_l_only",
                    choices=["lab_l_only"], help="Color handling for fusion.")
    ap.add_argument("--linear-fusion", action="store_true", default=False,
                    help="Decode sRGB->linear, fuse luminance in linear space, re-encode.")

    # Exposure-fusion style saturation weighting (DEFAULT: off)
    ap.add_argument("--sat-weight", action="store_true", default=False, help="Use saturation weighting.")
    ap.add_argument("--sat-gamma", type=float, default=1.0, help="Gamma/power for saturation weight (>=0).")

    # Luminance CLAHE (DEFAULT: off)
    ap.add_argument("--clahe-l", action="store_true", default=False, help="Apply CLAHE to fused luminance.")
    ap.add_argument("--clahe-clip", type=float, default=2.0, help="CLAHE clip limit.")
    ap.add_argument("--clahe-tiles", type=int, default=8, help="CLAHE tile grid size (N x N).")

    # GPU switch
    ap.add_argument("--gpu", action="store_true", help="Use CUDA acceleration via cv2.cuda if available.")

    # Physical pixel size for OME-TIFF
    ap.add_argument("--px", type=float, default=None, help="Physical pixel size X (micrometers).")
    ap.add_argument("--py", type=float, default=None, help="Physical pixel size Y (micrometers).")
    ap.add_argument("--save-ome", action="store_true", help="Also save OME-TIFF.")

    args = ap.parse_args()

    folder = Path(__file__).resolve().parent
    base_name = f"{folder.name}-deep"

    files = find_stack_images(folder, args.pattern)
    files = [f for f in files if f.suffix.lower() in IMG_EXTS]
    if len(files) < 2:
        print("Need at least two images in this folder.", file=sys.stderr)
        for f in files:
            print(" -", f.name)
        sys.exit(2)

    print(f"Found {len(files)} images:")
    for f in files:
        print(" -", f.name)

    images = [read_image(p) for p in files]
    if not ensure_same_channels(images):
        raise ValueError("Mixed channel counts in stack (e.g., grayscale + BGR). Make them consistent.")
    images = harmonize_sizes(images)

    # Illumination correction (default: none)
    if args.illum == "stack_median":
        flat = estimate_flatfield_stack_median(images, blur_sigma=args.illum_sigma)
        images = [apply_flatfield(im, flat) for im in images]
    elif args.illum == "per_slice":
        images = [apply_flatfield(im, estimate_flatfield_per_slice(im, blur_sigma=args.illum_sigma)) for im in images]

    # Fuse (with or without GPU)
    fused, depth = fuse_stack(
        images,
        color_mode=args.color_mode,
        focus=args.focus,
        levels=args.levels,
        sobel_ksize=args.sobel_ksize,
        linear_fusion=args.linear_fusion,
        use_sat_weight=args.sat_weight,
        sat_gamma=args.sat_gamma,
        clahe_l=args.clahe_l,
        clahe_clip=args.clahe_clip,
        clahe_tiles=args.clahe_tiles,
        use_gpu=args.gpu,
    )

    # Save PNG/TIFF with naming convention (BGR direct)
    out_img = folder / f"{base_name}.png"
    out_dep = folder / f"{base_name}_depth.tiff"
    cv2.imwrite(str(out_img), fused)
    cv2.imwrite(str(out_dep), depth)
    print(f"Saved:\n  {out_img}\n  {out_dep}")

    # Optional OME-TIFF (convert BGR->RGB)
    if args.save_ome:
        is_color = (fused.ndim == 3 and fused.shape[2] == 3)
        meta = {"axes": "YXC" if is_color else "YX"}
        if args.px is not None:
            meta["PhysicalSizeX"] = float(args.px); meta["PhysicalSizeXUnit"] = "µm"
        if args.py is not None:
            meta["PhysicalSizeY"] = float(args.py); meta["PhysicalSizeYUnit"] = "µm"

        ome_fused = folder / f"{base_name}.ome.tif"
        if is_color:
            rgb = cv2.cvtColor(fused, cv2.COLOR_BGR2RGB)
            tiff.imwrite(str(ome_fused), rgb, photometric="rgb", metadata=meta, bigtiff=False)
        else:
            tiff.imwrite(str(ome_fused), fused, photometric="minisblack", metadata=meta, bigtiff=False)

        ome_depth = folder / f"{base_name}_depth.ome.tif"
        meta_d = {"axes": "YX"}
        if args.px is not None:
            meta_d["PhysicalSizeX"] = float(args.px); meta_d["PhysicalSizeXUnit"] = "µm"
        if args.py is not None:
            meta_d["PhysicalSizeY"] = float(args.py); meta_d["PhysicalSizeYUnit"] = "µm"
        tiff.imwrite(str(ome_depth), depth, photometric="minisblack", metadata=meta_d, bigtiff=False)

        print(f"Saved OME-TIFF:\n  {ome_fused}\n  {ome_depth}")

    print("Done.")

if __name__ == "__main__":
    main()
