"""
绘画 CV 测量 —— 只出数，不出判断。

分层：ingest → segment → measures → Observation
铁律：measrues 层不出现"好/坏"；每条 Observation 带 region 和 method。

用法：
    python cvmeasure.py <图片或目录> [--out 输出.csv]
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, asdict

import cv2
import numpy as np


# ======================================================================
# 数据结构
# ======================================================================
@dataclass(frozen=True)
class Observation:
    key: str
    value: float
    unit: str = ""              # "" | "ratio" | "norm" | "px"
    region: tuple | None = None  # (x, y, w, h)
    conf: float = 1.0
    method: str = ""


@dataclass
class Analysis:
    file: str
    image_sha256: str
    params_version: str
    observations: list
    segment_ok: bool
    segment_method: str


PARAMS_VERSION = "v1"


# ======================================================================
# 1. ingest
# ======================================================================
def ingest(path):
    """读图 → RGB(白底合成) → 限长边 → LAB。返回 (rgb, lab, meta)"""
    raw = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"读不了: {path}")

    if img.ndim == 3 and img.shape[2] == 4:
        b, g, r, a = cv2.split(img)
        a = a.astype(np.float32) / 255.0
        rgb = cv2.merge([b, g, r]).astype(np.float32)
        rgb = rgb * a[..., None] + 255.0 * (1 - a[..., None])   # 合成到白底
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    elif img.ndim == 3:
        rgb = img[:, :, :3]
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    h, w = rgb.shape[:2]
    scale = min(1.0, 2000.0 / max(h, w))
    if scale < 1.0:
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    h, w = rgb.shape[:2]
    diag = math.hypot(w, h)
    lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB)
    meta = {"w": w, "h": h, "diag": diag}
    return rgb, lab, meta


# ======================================================================
# 2. segment —— 三级降级
# ======================================================================
def segment(rgb):
    """返回 (mask, ok, method)。拿不到干净前景就不硬算。"""
    h, w = rgb.shape[:2]
    total = h * w

    # ---- ① 纯色底：四角取色 ----
    corners = np.concatenate([
        rgb[0:20, 0:20].reshape(-1, 3), rgb[0:20, -20:].reshape(-1, 3),
        rgb[-20:, 0:20].reshape(-1, 3), rgb[-20:, -20:].reshape(-1, 3),
    ]).astype(np.float32)
    bg = np.median(corners, axis=0)
    dist = np.linalg.norm(rgb.astype(np.float32) - bg, axis=2)
    mask = (dist > 28).astype(np.uint8) * 255

    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    ratio = mask.sum() / 255 / total
    if 0.02 <= ratio <= 0.80:
        return mask, True, "corner_threshold"

    # ---- ② GrabCut 兜底 ----
    try:
        gc = np.zeros((h, w), np.uint8)
        gc[:] = cv2.GC_PR_BGD
        margin = max(3, int(min(h, w) * 0.04))
        m = np.full((h, w), cv2.GC_PR_FGD, np.uint8)
        m[:margin, :] = cv2.GC_BGD
        m[-margin:, :] = cv2.GC_BGD
        m[:, :margin] = cv2.GC_BGD
        m[:, -margin:] = cv2.GC_BGD
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(rgb, m, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
        m2 = np.where((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        m2 = cv2.morphologyEx(m2, cv2.MORPH_CLOSE, k, iterations=2)
        r2 = m2.sum() / 255 / total
        if 0.02 <= r2 <= 0.80:
            return m2, True, "grabcut"
    except Exception:
        pass

    # ---- ③ 都失败：不猜 ----
    return None, False, "failed"


# ======================================================================
# 3. measures
# ======================================================================
def m_contour(mask, meta):
    out = []
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return out
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    region = (x, y, w, h)

    area = cv2.contourArea(c)
    if area < 50:
        return out
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0
    out.append(Observation("silhouette.solidity", float(solidity), "ratio", region,
                           1.0, "cv2.convexHull/v1"))

    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    out.append(Observation("silhouette.vertex_norm",
                           float(len(approx) / max(peri, 1e-6) * 100), "norm", region,
                           1.0, "cv2.approxPolyDP/v1"))

    # Fourier 描述子：低频能量占比越高 = 外形越整
    pts = c.reshape(-1, 2).astype(np.complex128)
    z = pts[:, 0] + 1j * pts[:, 1]
    F = np.fft.fft(z)
    mag = np.abs(F)
    total = mag[1:].sum()
    if total > 1e-9:
        low = mag[1:11].sum()
        out.append(Observation("silhouette.fourier_lf_ratio",
                               float(low / total), "ratio", region, 1.0, "numpy.fft/v1"))

    hu = cv2.HuMoments(cv2.moments(c)).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    for i in range(1, 4):
        out.append(Observation(f"silhouette.hu{i}", float(hu_log[i]), "", region,
                               1.0, "cv2.HuMoments/v1"))

    out.append(Observation("silhouette.aspect", float(w) / max(h, 1), "ratio", region,
                           1.0, "bbox/v1"))
    out.append(Observation("silhouette.area_ratio", float(area / (meta["diag"] ** 2)),
                           "norm", region, 1.0, "contourArea/v1"))
    return out


def m_density(mask, meta):
    out = []
    ys, xs = np.nonzero(mask)
    if len(xs) < 100:
        return out
    idx = np.random.RandomState(0).choice(len(xs), size=min(3000, len(xs)), replace=False)
    pts = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(pts)
        d, _ = tree.query(pts, k=2)          # 第 0 个是自己
        nn = d[:, 1]
    except Exception:
        nn = np.array([np.nan])
    if np.isfinite(nn).all() and nn.mean() > 1e-9:
        out.append(Observation("density.nn_cv", float(nn.std() / nn.mean()),
                               "ratio", None, 0.9, "cKDTree/v1"))

    # 滑窗密度变异系数
    h, w = mask.shape
    win = max(16, min(h, w) // 8)
    dens = []
    for yy in range(0, h - win, win):
        for xx in range(0, w - win, win):
            dens.append(mask[yy:yy + win, xx:xx + win].sum() / 255.0 / (win * win))
    dens = np.array(dens)
    dens = dens[dens > 0.01]
    if len(dens) > 5 and dens.mean() > 1e-9:
        out.append(Observation("density.window_cv", float(dens.std() / dens.mean()),
                               "ratio", None, 1.0, "sliding/v1"))

    # 内部空洞
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    hole_area = 0.0
    if hier is not None:
        for i, hh in enumerate(hier[0]):
            if hh[3] != -1:                       # 有父轮廓 = 是洞
                hole_area += cv2.contourArea(cnts[i])
    out.append(Observation("density.hole_ratio",
                           float(hole_area / max(mask.sum() / 255.0, 1)),
                           "ratio", None, 1.0, "RETR_CCOMP/v1"))
    return out


def m_balance(mask, lab, meta):
    out = []
    ys, xs = np.nonzero(mask)
    if len(xs) < 100:
        return out
    L = lab[:, :, 0].astype(np.float32)
    ww = (255.0 - L) * (mask > 0)             # 越暗越"重"
    s = ww.sum()
    if s < 1e-6:
        return out
    cx = float((ww * np.arange(mask.shape[1])[None, :]).sum() / s)
    cy = float((ww * np.arange(mask.shape[0])[:, None]).sum() / s)
    ox = cx - mask.shape[1] / 2
    oy = cy - mask.shape[0] / 2
    off = math.hypot(ox, oy) / meta["diag"]
    out.append(Observation("balance.centroid_offset", float(off), "norm", None,
                           1.0, "luma_centroid/v1"))

    x, y, w, h = cv2.boundingRect(mask)
    out.append(Observation("balance.mass_ratio",
                           float((mask > 0).sum() / (mask.shape[0] * mask.shape[1])),
                           "ratio", (x, y, w, h), 1.0, "mask/v1"))

    # 主体质心到四个三分点的最近距离
    thirds = [(mask.shape[1] * t / 3, mask.shape[0] * t / 3) for t in (1, 2)]
    dmin = min(math.hypot(cx - tx, cy - ty) for tx in [t[0] for t in thirds]
               for ty in [t[1] for t in thirds])
    out.append(Observation("balance.thirds_dist", float(dmin / meta["diag"]), "norm",
                           None, 1.0, "thirds/v1"))

    # 负空间
    neg = (1 - (mask > 0)).astype(np.uint8)
    ncnts, _ = cv2.findContours(neg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if ncnts:
        areas = sorted((cv2.contourArea(c) for c in ncnts), reverse=True)
        tot = neg.sum()
        out.append(Observation("balance.neg_space_frac",
                               float(tot / (mask.shape[0] * mask.shape[1])),
                               "ratio", None, 1.0, "negspace/v1"))
        out.append(Observation("balance.neg_fragmentation",
                               float(len(areas)), "", None, 0.8, "negspace/v1"))
    return out


def m_color(rgb, lab, mask, meta):
    out = []
    m = mask > 0
    if m.sum() < 200:
        m = np.ones(mask.shape, bool)

    L = lab[:, :, 0][m].astype(np.float32)
    hist = cv2.calcHist([L], [0], None, [64], [0, 256]).flatten()
    hist = cv2.GaussianBlur(hist.reshape(-1, 1), (1, 5), 1.2).flatten()
    p = hist / max(hist.sum(), 1e-9)
    p = p[p > 1e-9]
    out.append(Observation("color.value_entropy", float(-(p * np.log2(p)).sum()),
                           "", None, 1.0, "hist_gauss/v1"))

    a = lab[:, :, 1][m].astype(np.float32) - 128
    b = lab[:, :, 2][m].astype(np.float32) - 128
    # ★ 色相圆周统计：必须先把每个像素归一化成【单位向量】再求合成向量长度，
    #   否则得到的是 LAB 里的绝对距离（0~180），不是 [0,1] 的平均合成长度。
    mag = np.hypot(a, b)
    keep = mag > 3.0                       # 只统计有色的像素（灰的没有色相）
    if keep.sum() > 100:
        ux = (a[keep] / mag[keep]).mean()
        uy = (b[keep] / mag[keep]).mean()
        R = math.hypot(ux, uy)
        R = min(max(R, 1e-6), 0.9999)
        cstd = math.sqrt(-2.0 * math.log(R))
    else:
        cstd = float("nan")                 # 几乎没有彩色像素，这个量无意义
    out.append(Observation("color.hue_circstd", float(cstd), "ratio", None,
                           0.8, "circstat/v1"))
    out.append(Observation("color.chroma_mean", float(np.hypot(a, b).mean()), "",
                           None, 1.0, "lab/v1"))

    # 主色簇数
    px = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB)[m].astype(np.float32)
    if len(px) > 5000:
        px = px[np.random.RandomState(0).choice(len(px), 5000, replace=False)]
    try:
        from sklearn.cluster import DBSCAN
        lab_db = DBSCAN(eps=12, min_samples=20).fit(px).labels_
        k = len(set(lab_db)) - (1 if -1 in lab_db else 0)
        out.append(Observation("color.palette_k", float(k), "", None, 0.8, "dbscan/v1"))
    except Exception:
        pass
    return out


def m_texture(rgb, mask, meta):
    out = []
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    m = (mask > 0)
    if m.sum() < 200:
        return out
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    out.append(Observation("texture.hf_energy",
                           float(np.abs(lap[m]).mean() / 255.0), "norm", None,
                           1.0, "laplacian/v1"))
    mean = cv2.blur(gray.astype(np.float32), (5, 5))
    sq = cv2.blur((gray.astype(np.float32) ** 2), (5, 5))
    var = np.clip(sq - mean ** 2, 0, None)
    out.append(Observation("texture.local_var_mean",
                           float(var[m].mean() / (255.0 ** 2)), "norm", None,
                           1.0, "boxfilter/v1"))
    return out


def m_symmetry(mask, meta):
    out = []
    f = mask.astype(np.float32) / 255.0
    flip = f[:, ::-1]
    denom = math.sqrt((f ** 2).sum() * (flip ** 2).sum())
    if denom < 1e-9:
        return out
    out.append(Observation("symmetry.mirror_score",
                           float((f * flip).sum() / denom), "ratio", None,
                           1.0, "flipcorr/v1"))
    return out


# ======================================================================
# 4. 汇总
# ======================================================================
def analyze(path):
    import hashlib
    with open(path, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()

    rgb, lab, meta = ingest(path)
    mask, ok, method = segment(rgb)
    obs = []
    if ok:
        # ★ 每个 measure 的形参不同，别用统一 dispatch 猜
        stages = [
            ("m_contour",  lambda: m_contour(mask, meta)),
            ("m_density",  lambda: m_density(mask, meta)),
            ("m_balance",  lambda: m_balance(mask, lab, meta)),
            ("m_texture",  lambda: m_texture(rgb, mask, meta)),
            ("m_symmetry", lambda: m_symmetry(mask, meta)),
            ("m_color",    lambda: m_color(rgb, lab, mask, meta)),
        ]
        for name, fn in stages:
            try:
                obs += fn()
            except Exception as e:
                obs.append(Observation(f"error.{name}", float("nan"), "", None,
                                       0.0, str(e)[:60]))
        # ★ 扩展因子（A/B/C 三族）
        try:
            from measures_extra import extra_contour, extra_density, extra_balance
            for name, fn in [
                ("extra_contour", lambda: extra_contour(mask, meta)),
                ("extra_density", lambda: extra_density(mask, meta)),
                ("extra_balance", lambda: extra_balance(mask, lab, meta)),
            ]:
                try:
                    obs += fn()
                except Exception as e:
                    obs.append(Observation(f"error.{name}", float("nan"), "", None,
                                           0.0, str(e)[:60]))
        except Exception as e:
            obs.append(Observation("error.extra_import", float("nan"), "", None,
                                   0.0, str(e)[:60]))
        # ★ 6:3:1 分布因子（疏密 / 颜色 / 饱和度 / 明度）
        try:
            from measures_631 import density631, color631, chroma631, value631
            for name, fn in [
                ("d631_density", lambda: density631(mask, meta)),
                ("d631_color",   lambda: color631(rgb, lab, mask, meta)),
                ("d631_chroma",  lambda: chroma631(lab, mask)),
                ("d631_value",   lambda: value631(lab, mask)),
            ]:
                try:
                    obs += fn()
                except Exception as e:
                    obs.append(Observation(f"error.{name}", float("nan"), "", None,
                                           0.0, str(e)[:60]))
        except Exception as e:
            obs.append(Observation("error.d631_import", float("nan"), "", None,
                                   0.0, str(e)[:60]))
        # ★ 主体白边预处理后的轮廓因子（带场景不失效）
        try:
            from preprocess_subject import contour_clean
            obs += contour_clean(rgb, mask, meta)
        except Exception as e:
            obs.append(Observation("error.subject", float("nan"), "", None,
                                   0.0, str(e)[:60]))
        # ★ 色彩统一性（各向异性 + 方向一致度）
        try:
            from measures_color_unity import emit_anisotropy
            obs = emit_anisotropy(obs, lab, mask)
        except Exception as e:
            obs.append(Observation("error.unity", float("nan"), "", None,
                                   0.0, str(e)[:60]))
        # ★ 明度-纯度单调关系（用户提出的核心因子）
        try:
            from measures_value_chroma import emit_value_chroma
            obs = emit_value_chroma(obs, lab, mask)
        except Exception as e:
            obs.append(Observation("error.vc", float("nan"), "", None,
                                   0.0, str(e)[:60]))
        # ★ 四量两两关系 + 空间关系
        try:
            from measures_relations import emit_relations
            obs = emit_relations(obs, rgb, lab, mask)
        except Exception as e:
            obs.append(Observation("error.rel", float("nan"), "", None,
                                   0.0, str(e)[:60]))
    return Analysis(
        file=os.path.basename(path), image_sha256=sha,
        params_version=PARAMS_VERSION,
        observations=[asdict(o) for o in obs],
        segment_ok=ok, segment_method=method,
    )


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    target = sys.argv[1]
    out_csv = sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "--out" else None

    if os.path.isdir(target):
        files = [os.path.join(target, f) for f in sorted(os.listdir(target))
                 if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
    else:
        files = [target]

    rows, results = [], []
    for f in files:
        try:
            a = analyze(f)
        except Exception as e:
            print(f"  ✗ {os.path.basename(f)}: {type(e).__name__} {e}")
            continue
        results.append(asdict(a))
        row = {"file": a.file, "segment": a.segment_method,
               "ok": a.segment_ok}
        for o in a.observations:
            row[o["key"]] = round(o["value"], 5) if isinstance(o["value"], (int, float)) else None
        rows.append(row)
        print(f"  {'✓' if a.segment_ok else '✗'} {a.file:44s} {a.segment_method:18s} "
              f"{len(a.observations)} 项")

    print(f"\n共 {len(rows)} 张")
    if rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        if out_csv:
            df.to_csv(out_csv, index=False, encoding="utf-8-sig")
            print(f"CSV -> {out_csv}")
        json_path = (out_csv or "out.csv").replace(".csv", "_raw.json")
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=1)
        print(f"原始 Observation -> {json_path}")
        cols = [c for c in df.columns if c.startswith(("silhouette.", "contour.", "density.",
                                                       "balance.", "color.",
                                                       "texture.", "symmetry."))]
        print(f"\n指标 {len(cols)} 项：")
        for c in cols:
            v = df[c].dropna()
            if len(v):
                print(f"  {c:32s} n={len(v):3d}  {v.min():8.4f} ~ {v.max():8.4f}  均值 {v.mean():8.4f}")


if __name__ == "__main__":
    main()
