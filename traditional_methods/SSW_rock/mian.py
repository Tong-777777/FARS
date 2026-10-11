# This script is based on "A sliding window method considering image shadow to detect Mars rock from MRO HiRISE datasets."

from __future__ import annotations

import os.path
import math
import numpy as np
import cv2
import rasterio
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from typing import Tuple, List
from skimage.measure import shannon_entropy
import pandas as pd
import json
from typing import Dict, Any, Optional

# Parameters (based on HiRISE metadata)
SUB_SOLAR_AZIMUTH: float = 333.4
# NORTH_SOLAR_AZIMUTH: float = 99

# SOLAR_AZIMUTH: float = SUB_SOLAR_AZIMUTH + NORTH_SOLAR_AZIMUTH

# Solar azimuth (degrees)
INCIDENCE_ANGLE: int = 44  # Solar incidence angle (degrees)
STEP_RATIO: float = 1 / 8  # Sliding window step length ratio
GRAY_DIFF_THRESHOLD: int = 30  # Threshold for mean intensity difference

### 1. Enhancing image edges using high-pass filtering. "we tried to utilize high-pass filtering to obtain partial edge energy information for enhancing HiRISE images" in Section 2.1, Page 3. ###
def enhance_edges(image: np.ndarray, sigma: float = 1.0, alpha: float = 1.5) -> np.ndarray:
    """
    Args:
        image (np.ndarray): Input single-band grayscale image.
        sigma (float): Standard deviation for Gaussian blur (controls smoothing).
        alpha (float): Edge enhancement factor.

    Returns:
        np.ndarray: Edge-enhanced image as uint8 array.
    """
    lowpass = ndimage.gaussian_filter(image.astype(float), sigma)
    highpass = image - lowpass
    enhanced = np.clip(image + alpha * highpass, 0, 255).astype(np.uint8)
    return enhanced

### 2. Extracting shadows of rocks. "The shadow areas are then obtained by image segmentation using the OTSU method." ###
def extract_shadows(image: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Args:
        image (np.ndarray): Enhanced grayscale image.

    Returns:
        tuple[np.ndarray, float]:
            - Binary shadow mask (0 = background, 1 = shadow).
            - Optimal threshold value computed by Otsu method.
    """
    thresh: float = threshold_otsu(image)
    shadow_mask: np.ndarray = (image < thresh).astype(np.uint8)
    return shadow_mask, thresh

### 3. Getting the minimum bounding rectangle aligned with solar_azimuth. ###
def get_minimum_bounding_rectangle(region, solar_azimuth: float) -> Tuple[
    Tuple[float, float], Tuple[float, float], float, np.ndarray]:
    """
    Returns:
      (cx, cy)       : rectangle center in image coordinates (float, float)
      (w, h)         : width (along solar azimuth) and height (perp) in pixels
      angle          : angle in degrees (== solar_azimuth)
      corners (4x2)  : rectangle corner coordinates in original image coords,
                       order: [bottom-left, bottom-right, top-right, top-left] in that (x,y) sense
    Notes:
      - region.coords comes as (row, col). We convert to (x=col, y=row).
      - solar_azimuth is treated in image coordinates: x to right, y down, direction vector u=(cos, sin).
    """
    pts = region.coords[:, ::-1].astype(np.float64)  # (x, y)

    theta = np.deg2rad(solar_azimuth)
    # Basis vectors: u along solar azimuth, v perpendicular (90deg CCW of u)
    u = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    v = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float64)

    # Project all points onto (u, v)
    proj_u = pts.dot(u)  # scalar projections onto u
    proj_v = pts.dot(v)  # scalar projections onto v

    x_min, x_max = proj_u.min(), proj_u.max()
    y_min, y_max = proj_v.min(), proj_v.max()

    w = x_max - x_min  # length along u
    h = y_max - y_min  # length along v

    # center in projected coords -> map back to original coords
    center_proj = np.array([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0])
    center = center_proj[0] * u + center_proj[1] * v
    cx, cy = float(center[0]), float(center[1])

    # Construct 4 corners in projected coords then map back:
    corners_proj = np.array([
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max]
    ], dtype=np.float64)  # shape (4,2)

    # Map corners back: corner = proj_u * u + proj_v * v
    corners = corners_proj[:, 0:1] * u.reshape(1, 2) + corners_proj[:, 1:2] * v.reshape(1, 2)
    # corners is (4,2) with columns [x,y]
    return (cx, cy), (w, h), float(solar_azimuth), corners

### 4. Determing the centers of initial windows. Section 2.3, Page 4. ###
def pick_rock_side_edge_mid_geom(corners: np.ndarray, u: np.ndarray, pick: str = "max") -> np.ndarray:
    """
    Args:

        corners: (4,2) shadow corners in (x,y)
        u: illumination direction unit vector in image coords
        pick="max": choose edge with maximum projection onto u  (rock-side, if +u points shadow->rock)
        pick="min": choose edge with minimum projection onto u  (rock-side, if +u points rock->shadow)
    """
    proj = corners @ u  # (4,)

    order = np.argsort(proj)
    if pick == "max":
        edge = corners[order[-2:], :]   # two corners with the largest projection -> same edge
    else:
        edge = corners[order[:2], :]    # two corners with the smallest projection

    return edge.mean(axis=0)            # midpoint of that edge

### 5. Determining the termination windows using sliding windows
def sliding_window_rotated(image: np.ndarray,
                           init_x: float,
                           init_y: float,
                           step: float,
                           u: np.ndarray,
                           v: np.ndarray,
                           half_along: float,
                           half_perp: float) -> List[Tuple[float, float]]:
    """
    Sliding window using rotated-rectangle mean (paper-aligned window).
    Stop rules keep your original logic.
    """
    m_prev = rotated_rect_mean(image, (init_x, init_y), half_along, half_perp, u, v)

    if not np.isfinite(m_prev):
        return []

    positions: List[Tuple[float, float]] = [(init_x, init_y)]

    for i in range(1, 500):
        xi = init_x + u[0] * step * i
        yi = init_y + u[1] * step * i

        m_cur = rotated_rect_mean(image, (xi, yi), half_along, half_perp, u, v)

        if not np.isfinite(m_cur):
            break

        positions.append((xi, yi))

        # The termination condiction is that the grayscale mean of the current window is less than the mean of the previous window
        if m_cur < m_prev:
            break

        # 1) When the window is moved i(i≥4) times.
        # 2) The difference between the mean of the current window and the previous window is greater than a threshold (30).
        if i >= 4 and (m_cur - m_prev) < GRAY_DIFF_THRESHOLD:
            break

        m_prev = m_cur

    return positions

def rotated_rect_mean(image: np.ndarray,
                      center_xy: Tuple[float, float],
                      half_along: float,
                      half_perp: float,
                      u: np.ndarray,
                      v: np.ndarray) -> float:
    """
    Mean intensity inside a rotated rectangle centered at center_xy.
    Rectangle axes: u (along illumination), v (perp).
    """
    cx, cy = center_xy
    # corners for bbox
    ua = u * half_along
    vb = v * half_perp
    corners = np.stack([
        [cx - ua[0] - vb[0], cy - ua[1] - vb[1]],
        [cx + ua[0] - vb[0], cy + ua[1] - vb[1]],
        [cx + ua[0] + vb[0], cy + ua[1] + vb[1]],
        [cx - ua[0] + vb[0], cy - ua[1] + vb[1]],
    ], axis=0)

    xs = corners[:, 0]; ys = corners[:, 1]
    x0 = int(np.floor(xs.min())); x1 = int(np.ceil(xs.max()))
    y0 = int(np.floor(ys.min())); y1 = int(np.ceil(ys.max()))

    H, W = image.shape
    if x0 < 0 or y0 < 0 or x1 >= W or y1 >= H:
        return np.nan

    patch = image[y0:y1 + 1, x0:x1 + 1].astype(np.float64)

    yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    dx = xx - cx
    dy = yy - cy
    proj_u = dx * u[0] + dy * u[1]
    proj_v = dx * v[0] + dy * v[1]

    m = (np.abs(proj_u) <= half_along) & (np.abs(proj_v) <= half_perp)
    if not np.any(m):
        return np.nan
    return float(patch[m].mean())

def draw_rotated_window(img: np.ndarray,
                        center_xy: Tuple[int, int],
                        half_along: float,
                        half_perp: float,
                        u: np.ndarray,
                        v: np.ndarray,
                        color: Tuple[int, int, int],
                        thickness: int = 1) -> None:
    cx, cy = float(center_xy[0]), float(center_xy[1])
    ua = u * half_along
    vb = v * half_perp
    corners = np.stack([
        [cx - ua[0] - vb[0], cy - ua[1] - vb[1]],
        [cx + ua[0] - vb[0], cy + ua[1] - vb[1]],
        [cx + ua[0] + vb[0], cy + ua[1] + vb[1]],
        [cx - ua[0] + vb[0], cy - ua[1] + vb[1]],
    ], axis=0)
    poly = np.int32(np.round(corners))
    cv2.polylines(img, [poly], True, color, thickness, lineType=cv2.LINE_AA)

def detect_rocks(
        image: np.ndarray,
        shadow_mask: np.ndarray,
        output_path: str,
        file_name: str,
        slide_path: str,
        save_debug: bool = True,
) -> Tuple[List[dict], np.ndarray, Dict[str, Any]]:
    results: List[dict] = []
    debug: Dict[str, Any] = {
        "mbr_polys": [],     # list[np.ndarray shape (4,2) int32]
        "windows": [],       # list[dict]  start/end
    }

    labeled = label(shadow_mask)
    props = regionprops(labeled)

    rock_mask = np.zeros_like(image, dtype=np.uint8)

    vis = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    base_img = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    overlay = base_img.copy()
    slide_img = base_img.copy()

    # illumination basis (constant per image/tile)
    theta = np.deg2rad(SUB_SOLAR_AZIMUTH)
    u = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    v = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float64)

    for region in props:

        if region.area < 3:
            continue

        (cx, cy), (w, h), angle, corners = get_minimum_bounding_rectangle(region, SUB_SOLAR_AZIMUTH)
        poly = np.round(corners).astype(np.int32)

        debug["mbr_polys"].append(poly)

        cv2.fillPoly(overlay, [poly], (0, 0, 255))
        cv2.polylines(vis_bgr, [poly], isClosed=True, color=(0, 0, 255), thickness=1)

        shadow_width = float(h)
        r0 = shadow_width / 2.0
        if r0 <= 1:
            continue

        half_along = r0 / 2.0
        half_perp  = r0
        step = r0 * STEP_RATIO

        rock_edge_mid = pick_rock_side_edge_mid_geom(corners, u, pick="max")
        init_center = rock_edge_mid - u * half_along
        init_x, init_y = float(init_center[0]), float(init_center[1])

        positions = sliding_window_rotated(image, init_x, init_y, step, u, v, half_along, half_perp)
        if len(positions) < 2:
            continue

        start = np.array(positions[0], dtype=np.float64)
        end   = np.array(positions[-1], dtype=np.float64)

        Ps = start - u * half_along
        Pe = end   + u * half_along
        Lx = float(np.dot(Pe - Ps, u))
        Lx_abs = abs(Lx)

        axis_a = Lx_abs / 2.0            # semi-major axis a
        axis_b = r0                      # semi-minor axis b = radius

        if axis_a <= 0 or axis_b <= 0:
            continue

        center_xy = (Ps + Pe) / 2.0
        center = (float(center_xy[0]), float(center_xy[1]))

        ellipse_angle = (np.degrees(np.arctan2(u[1], u[0])) + 360.0) % 360.0

        shadow_length = abs(float(np.dot(end - start, u)))
        h_est = shadow_length / np.tan(np.deg2rad(INCIDENCE_ANGLE))
        if axis_a > axis_b:
            ha = (shadow_length + axis_a - axis_b) / np.tan(np.deg2rad(INCIDENCE_ANGLE))
            height = (h_est + ha) / 2.0
        else:
            height = h_est

        debug["windows"].append({
            "start": (float(start[0]), float(start[1])),
            "end":   (float(end[0]), float(end[1])),
            "half_along": float(half_along),
            "half_perp":  float(half_perp),
            "u": (float(u[0]), float(u[1])),
            "v": (float(v[0]), float(v[1])),
        })

        if save_debug:
            draw_rotated_window(slide_img, (start[0], start[1]), half_along, half_perp, u, v, (0, 255, 0), 1)
            draw_rotated_window(slide_img, (end[0], end[1]),   half_along, half_perp, u, v, (0, 255, 255), 1)

        center_coords = (int(round(center[0])), int(round(center[1])))
        axes = (int(round(axis_a)), int(round(axis_b)))
        temp_mask = np.zeros_like(image, dtype=np.uint8)
        cv2.ellipse(temp_mask, center_coords, axes, ellipse_angle, 0, 360, 1, thickness=-1)
        if not np.any(temp_mask == 1):
            continue

        cv2.ellipse(rock_mask, center_coords, axes, ellipse_angle, 0, 360, 1, thickness=-1)

        results.append({
            "x": center[0],
            "y": center[1],
            "axis_a": axis_a,
            "axis_b": axis_b,
            "angle": float(ellipse_angle),
            "height": height,

            "mbr_poly": poly.copy(),  # (4,2) int32
            "win": {
                "start": (float(start[0]), float(start[1])),
                "end": (float(end[0]), float(end[1])),
                "half_along": float(half_along),
                "half_perp": float(half_perp),
                "u": (float(u[0]), float(u[1])),
                "v": (float(v[0]), float(v[1])),
            }
        })

    cv2.addWeighted(overlay, 0.25, vis_bgr, 1 - 0.25, 0, vis_bgr)

    if save_debug:
        cv2.imwrite(os.path.join(output_path, f"{file_name}.png"), vis_bgr)
        cv2.imwrite(os.path.join(slide_path, f"{file_name}.png"), slide_img)

    debug["vis_bgr"] = vis_bgr
    debug["slide_img"] = slide_img
    return results, rock_mask, debug

def iter_tiles(H: int, W: int, tile: int, overlap: int):

    step = max(1, tile - overlap)
    xs = list(range(0, max(1, W - tile + 1), step))
    ys = list(range(0, max(1, H - tile + 1), step))
    if xs[-1] != W - tile: xs.append(max(0, W - tile))
    if ys[-1] != H - tile: ys.append(max(0, H - tile))
    for y0 in ys:
        for x0 in xs:
            yield x0, y0, min(W, x0 + tile), min(H, y0 + tile)

def tiled_collect_global(image: np.ndarray,
                         tile: int = 150,
                         overlap: int = 50,
                         halo: int | None = None,
                         pad_mode: str = "reflect"):
    """
    """
    if halo is None:
        halo = overlap

    H, W = image.shape

    p = halo
    img_pad = np.pad(image, ((p, p), (p, p)), mode=pad_mode)

    shadow_global = np.zeros((H, W), np.uint8)

    ellipses_global = []

    for x0, y0, x1, y1 in iter_tiles(H, W, tile, overlap):

        ox = overlap // 2
        oy = overlap // 2
        vx0 = x0 if x0 == 0 else x0 + ox
        vy0 = y0 if y0 == 0 else y0 + oy
        vx1 = x1 if x1 == W else x1 - ox
        vy1 = y1 if y1 == H else y1 - oy

        x0p, y0p = x0 + p, y0 + p
        x1p, y1p = x1 + p, y1 + p

        xe0, ye0 = x0p - halo, y0p - halo
        xe1, ye1 = x1p + halo, y1p + halo

        tile_img = img_pad[ye0:ye1, xe0:xe1]

        enhanced = enhance_edges(tile_img, sigma=2.0, alpha=1.8)
        shadow_mask01, _ = extract_shadows(enhanced)

        shift_x = x0 - halo
        shift_y = y0 - halo

        def paste_valid(dst, src01, shift_x, shift_y, vx0, vx1, vy0, vy1):
            th, tw = src01.shape
            dx0, dy0, dx1, dy1 = int(vx0), int(vy0), int(vx1), int(vy1)
            sx0, sy0, sx1, sy1 = dx0 - shift_x, dy0 - shift_y, dx1 - shift_x, dy1 - shift_y

            if sx0 < 0: dx0 += -sx0; sx0 = 0
            if sy0 < 0: dy0 += -sy0; sy0 = 0
            if sx1 > tw: dx1 -= (sx1 - tw); sx1 = tw
            if sy1 > th: dy1 -= (sy1 - th); sy1 = th

            Hh, Ww = dst.shape
            if dx0 < 0: sx0 += -dx0; dx0 = 0
            if dy0 < 0: sy0 += -dy0; dy0 = 0
            if dx1 > Ww: sx1 -= (dx1 - Ww); dx1 = Ww
            if dy1 > Hh: sy1 -= (dy1 - Hh); dy1 = Hh

            if dx1 <= dx0 or dy1 <= dy0 or sx1 <= sx0 or sy1 <= sy0:
                return

            src255 = (src01[sy0:sy1, sx0:sx1].astype(np.uint8) * 255)
            dst_patch = dst[dy0:dy1, dx0:dx1]
            dst[dy0:dy1, dx0:dx1] = np.maximum(dst_patch, src255)

        paste_valid(shadow_global, shadow_mask01, shift_x, shift_y, vx0, vx1, vy0, vy1)

        rocks, _, _ = detect_rocks(
            tile_img, shadow_mask01,
            output_path=".", file_name="tmp", slide_path=".",
            save_debug=False
        )

        for r in rocks:
            rr = dict(r)

            gx = rr["x"] + shift_x
            gy = rr["y"] + shift_y

            if not (vx0 <= gx < vx1 and vy0 <= gy < vy1):
                continue

            rr["x"] = gx
            rr["y"] = gy

            if "mbr_poly" in rr:
                poly = rr["mbr_poly"].astype(np.int32).copy()
                poly[:, 0] += int(round(shift_x))
                poly[:, 1] += int(round(shift_y))
                rr["mbr_poly"] = poly

            if "win" in rr:
                wrec = dict(rr["win"])
                sx, sy = wrec["start"]
                ex, ey = wrec["end"]
                wrec["start"] = (sx + shift_x, sy + shift_y)
                wrec["end"]   = (ex + shift_x, ey + shift_y)
                rr["win"] = wrec

            if 0 <= rr["x"] < W and 0 <= rr["y"] < H:
                ellipses_global.append(rr)

    return ellipses_global, shadow_global

def ellipse_area(e: dict) -> float:

    return float(math.pi * e["axis_a"] * e["axis_b"])

def ellipse_bbox(e: dict):

    cx, cy = float(e["x"]), float(e["y"])
    a, b = float(e["axis_a"]), float(e["axis_b"])
    ang = math.radians(float(e["angle"]) % 360.0)
    ca, sa = math.cos(ang), math.sin(ang)
    hx = math.sqrt((a * ca) ** 2 + (b * sa) ** 2)
    hy = math.sqrt((a * sa) ** 2 + (b * ca) ** 2)
    return cx - hx, cy - hy, cx + hx, cy + hy

def overlap_area_px(e1: dict, e2: dict, max_patch=250*250) -> int:

    b1 = ellipse_bbox(e1); b2 = ellipse_bbox(e2)
    ix0, iy0 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix1, iy1 = min(b1[2], b2[2]), min(b1[3], b2[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0

    x0, y0 = int(math.floor(ix0)), int(math.floor(iy0))
    x1, y1 = int(math.ceil(ix1)),  int(math.ceil(iy1))
    w, h = x1 - x0 + 1, y1 - y0 + 1
    if w <= 0 or h <= 0:
        return 0

    scale = 1
    if w * h > max_patch:
        scale = max(2, int(math.ceil(math.sqrt((w*h)/max_patch))))

    W = int(math.ceil(w/scale)); H = int(math.ceil(h/scale))
    m1 = np.zeros((H, W), np.uint8)
    m2 = np.zeros((H, W), np.uint8)

    def draw(m, e):
        cx = (float(e["x"]) - x0) / scale
        cy = (float(e["y"]) - y0) / scale
        a  = float(e["axis_a"]) / scale
        b  = float(e["axis_b"]) / scale
        ang = float(e["angle"]) % 360.0
        cv2.ellipse(m, (int(round(cx)), int(round(cy))),
                    (max(1, int(round(a))), max(1, int(round(b)))),
                    ang, 0, 360, 1, -1)

    draw(m1, e1); draw(m2, e2)
    return int(np.logical_and(m1==1, m2==1).sum() * (scale*scale))

### Optimization. Fitted ellipses with very large areas have a low probability of being rocks and also need to be removed. Section 2.5, Page 7. ###
def optimize_ellipses(ellipses: list[dict], max_area: float | None = None) -> list[dict]:

    tmp = []
    for e in ellipses:
        A = ellipse_area(e)
        if (max_area is not None) and (A > max_area):
            continue
        ee = dict(e); ee["_A"] = A; ee["_bbox"] = ellipse_bbox(ee)
        tmp.append(ee)

    tmp.sort(key=lambda x: x["_A"], reverse=True)
    kept = []

    for e in tmp:
        drop = False
        for k in kept:
            b1 = e["_bbox"]; b2 = k["_bbox"]
            if b1[2] < b2[0] or b2[2] < b1[0] or b1[3] < b2[1] or b2[3] < b1[1]:
                continue

            A1, A2 = e["_A"], k["_A"]
            if abs(A1 - A2) < 1e-6 * max(A1, A2):
                drop = True
                break

            ov = overlap_area_px(e, k)
            smaller = min(A1, A2)
            if smaller > 0 and (ov / smaller) > 0.5:
                if A1 <= A2:
                    drop = True
                    break
                else:
                    kept.remove(k)

        if not drop:
            kept.append(e)

    for e in kept:
        e.pop("_A", None)
        e.pop("_bbox", None)
    return kept

def normalize_gray_to_bgr(image: np.ndarray) -> np.ndarray:
    base = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

def draw_global_vis_bgr(image: np.ndarray, mbr_polys: list[np.ndarray]) -> np.ndarray:
    vis_bgr = normalize_gray_to_bgr(image)
    overlay = vis_bgr.copy()
    for poly in mbr_polys:
        cv2.fillPoly(overlay, [poly], (0, 0, 255))
        cv2.polylines(vis_bgr, [poly], True, (0, 0, 255), 1)
    cv2.addWeighted(overlay, 0.25, vis_bgr, 0.75, 0, vis_bgr)
    return vis_bgr

def draw_global_slide_img(image: np.ndarray, windows: list[dict]) -> np.ndarray:
    slide_img = normalize_gray_to_bgr(image)
    for wrec in windows:
        u = np.array(wrec["u"], dtype=np.float64)
        v = np.array(wrec["v"], dtype=np.float64)
        ha = float(wrec["half_along"])
        hp = float(wrec["half_perp"])
        s = wrec["start"]
        e = wrec["end"]
        draw_rotated_window(slide_img, s, ha, hp, u, v, (0, 255, 0), 1)       # start → green
        draw_rotated_window(slide_img, e, ha, hp, u, v, (0, 255, 255), 1)     # end → yellow
    return slide_img

def draw_global_eli_output(image: np.ndarray, ellipses: list[dict]) -> np.ndarray:
    eli = normalize_gray_to_bgr(image)
    for e in ellipses:
        center = (int(round(e["x"])), int(round(e["y"])))
        axes   = (int(round(e["axis_a"])), int(round(e["axis_b"])))
        ang    = float(e["angle"]) % 360.0
        cv2.ellipse(eli, center, axes, ang, 0, 360, (0, 0, 255), 1)
    return eli

def draw_global_rock_mask(shape_hw: Tuple[int, int], ellipses: list[dict]) -> np.ndarray:
    H, W = shape_hw
    mask = np.zeros((H, W), np.uint8)
    for e in ellipses:
        center = (int(round(e["x"])), int(round(e["y"])))
        axes   = (int(round(e["axis_a"])), int(round(e["axis_b"])))
        ang    = float(e["angle"]) % 360.0
        cv2.ellipse(mask, center, axes, ang, 0, 360, 1, thickness=-1)
    return mask

def make_overlay_img(image: np.ndarray, rock_mask: np.ndarray) -> np.ndarray:
    base = normalize_gray_to_bgr(image)
    red = np.zeros_like(base); red[:, :, 2] = 255
    out = base.copy()
    out[rock_mask == 1] = red[rock_mask == 1]
    return out

def make_shadow_overlay(image: np.ndarray, shadow_mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:

    base = normalize_gray_to_bgr(image)
    red  = np.zeros_like(base)
    red[:, :, 2] = 255

    blended = cv2.addWeighted(base, 1.0 - alpha, red, alpha, 0)
    out = base.copy()

    m = shadow_mask > 0
    out[m] = blended[m]
    return out

def run_tiled_whole(image: np.ndarray,
                    tile_size: int = 150,
                    overlap_ratio: float = 0.3,
                    max_area: float | None = None):
    """
    150x150 tiled
    Return：
      rock_mask, overlay_img, eli_img, vis_bgr, slide_img
    """
    overlap = int(round(tile_size * overlap_ratio))

    ellipses_raw, shadow_mask = tiled_collect_global(image, tile=150, overlap=overlap, halo=None)

    final_ellipses = optimize_ellipses(ellipses_raw, max_area=None)

    rock_mask = draw_global_rock_mask(image.shape, final_ellipses)
    overlay_img = make_overlay_img(image, rock_mask)
    eli_img = draw_global_eli_output(image, final_ellipses)

    mbr_polys = [e["mbr_poly"] for e in final_ellipses if "mbr_poly" in e]
    windows = [e["win"] for e in final_ellipses if "win" in e]

    vis_bgr = draw_global_vis_bgr(image, mbr_polys)
    slide_img = draw_global_slide_img(image, windows)

    return rock_mask, overlay_img, eli_img, vis_bgr, slide_img, shadow_mask

def make_shadow_overlay(image: np.ndarray, shadow_mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:

    base = normalize_gray_to_bgr(image)  # 原图归一化到0..255并转BGR
    red = np.zeros_like(base)
    red[:, :, 2] = 255  # BGR里红色通道是2

    blended = cv2.addWeighted(base, 1.0 - alpha, red, alpha, 0)
    out = base.copy()

    m = shadow_mask > 0
    out[m] = blended[m]
    return out



if __name__ == '__main__':

    # image_number = 'TRA_000828_2495_RED'
    #
    # base_path = r'F:\mars_rock\HiR\dataset'

    base_path = r'F:\mars_rock\unsupervised_mars_rock_detection\benchmark_methods\SSW-ROCK\training_data'

    image_number = 'TRA_000894_2475'

    # Read HiRISE images
    file_path = r'{}\{}\img'.format(base_path, image_number)
    rect_path = r'{}\{}\shadow_rect_output'.format(base_path, image_number)
    shadow_path = r'{}\{}\shadow_output'.format(base_path, image_number)
    eli_path = r'{}\{}\eli_output'.format(base_path, image_number)
    mask_path = r'{}\{}\mask_output'.format(base_path, image_number)
    slide_start_path = r'{}\{}\slide_start_output'.format(base_path, image_number)
    overlay_path = r'{}\{}\overlay_output'.format(base_path, image_number)

    os.makedirs(rect_path, exist_ok=True)
    os.makedirs(eli_path, exist_ok=True)
    os.makedirs(shadow_path, exist_ok=True)
    os.makedirs(mask_path, exist_ok=True)
    os.makedirs(overlay_path, exist_ok=True)
    os.makedirs(slide_start_path, exist_ok=True)

    files = os.listdir(file_path)

    for file in files:
        if file.endswith('tif'):
            with rasterio.open(os.path.join(file_path, file)) as src:
                # with rasterio.open(r'output.tif') as src:
                image = src.read(1)

            H, W = image.shape

            point_mask = np.zeros_like(image, dtype=np.uint8)

            file_name = file.split('.')[0]
            file_type = file.split('.')[1]

            all_rocks = []

            # --- Detect rocks ---
            rock_mask, overlay_img, eli_img, vis_bgr, slide_img, shadow_mask = run_tiled_whole(
                image=image,
                tile_size=150,
                overlap_ratio=0.3,
                max_area=None
            )

            cv2.imwrite(os.path.join(mask_path, f"{file_name}_mask.png"), rock_mask)

            cv2.imwrite(os.path.join(overlay_path, f"{file_name}_overlay.png"), overlay_img)

            cv2.imwrite(os.path.join(eli_path, f"{file_name}.png"), eli_img)

            cv2.imwrite(os.path.join(rect_path, f"{file_name}.png"), vis_bgr)

            shadow_overlay = make_shadow_overlay(image, shadow_mask, alpha=0.5)
            cv2.imwrite(os.path.join(shadow_path, f"{file_name}_shadow_overlay.png"), shadow_overlay)

            cv2.imwrite(os.path.join(slide_start_path, f"{file_name}.png"), slide_img)

