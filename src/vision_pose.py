"""Vision-only virtual-target estimation for the multiball sphere-in-box task.

The policy's point cloud is produced by the same fixed depth+RGB camera
(``cam_pc``) as the estimator, so the servo targets and the servo error are
computed from exactly the observation the policy sees -- no ``env.data``
object pose, no ``env.sphere_pos()`` and no ``env.box_center`` in the default
(``vision``) path.

Two virtual targets are estimated:

1. **ball position** -- the GREEN target sphere's centre, from hue
   segmentation + a robust fixed-radius (35 mm) sphere fit.  The fit uses the
   known radius, so the visible surface curvature pins the centre; a free
   centroid would be biased toward the camera by ~R/2.
2. **pre-drop position** -- the ball-centre pose directly above the box
   opening, at the expert's carry height.  The box opening centre and rim
   height are estimated from the brown box wall/top points and frozen per
   episode (the box is a static fixture), with the scene fixture constants as
   a validity-checked fallback.

The two targets are ball-centre poses.  The policy commands the finger site,
which sits ``EE_TO_BALL_Z`` above the centre of a held ball, so the caller
converts an EE pose to the implied ball-centre pose before forming the error.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# ----------------------------------------------------------------------
# scene geometry (must match multiball/sphere_box_multiball.xml)
# ----------------------------------------------------------------------
BALL_RADIUS = 0.035
BALL_RGB = np.array([0.15, 0.75, 0.20], dtype=np.float64) * 255.0
BOX_RGB = np.array([0.60, 0.45, 0.15], dtype=np.float64) * 255.0

TABLE_SURFACE_Z = 0.19
BOX_CENTER = np.array([0.30, 0.15], dtype=np.float64)
BOX_RIM_Z = TABLE_SURFACE_Z + 0.08          # 0.27
# Expert carry height (finger_site) and the finger_site -> held-ball-centre
# vertical offset.  The expert's CARRY_Z is 0.365 in the collector.
CARRY_SITE_Z = 0.365
EE_TO_BALL_Z = 0.011

# workspace sanity bounds for the ball estimate (metres)
BALL_XYZ_MIN = np.array([0.04, -0.32, 0.16])
BALL_XYZ_MAX = np.array([0.55, 0.44, 0.60])


# ----------------------------------------------------------------------
# colour segmentation
# ----------------------------------------------------------------------
def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """(..., 3) RGB in 0-255 -> (..., 3) HSV with H in [0, 1)."""
    x = np.asarray(rgb, dtype=np.float64) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx = np.max(x, axis=-1)
    mn = np.min(x, axis=-1)
    d = mx - mn
    h = np.zeros_like(mx)
    m = d > 1e-9
    rm = m & (mx == r)
    gm = m & (mx == g)
    bm = m & (mx == b)
    h[rm] = ((g[rm] - b[rm]) / d[rm]) % 6.0
    h[gm] = (b[gm] - r[gm]) / d[gm] + 2.0
    h[bm] = (r[bm] - g[bm]) / d[bm] + 4.0
    h = h / 6.0
    s = np.where(mx > 1e-9, d / np.maximum(mx, 1e-9), 0.0)
    return np.stack([h, s, mx], axis=-1)


def hue_of(rgb: np.ndarray) -> float:
    return float(rgb_to_hsv(np.asarray(rgb, dtype=np.float64))[..., 0])


def segment_color(points: np.ndarray, rgb: np.ndarray,
                  hue_tol: float = 0.07, s_min: float = 0.30,
                  v_min: float = 0.08) -> np.ndarray:
    """Boolean mask of points matching ``rgb`` (xyz+rgb, RGB 0-255).

    Points whose ghost-flag channel (column 6, if present) is set are never
    treated as real scene points.
    """
    if points.shape[1] >= 7:
        real = points[:, 6] < 0.5
    else:
        real = np.ones(len(points), dtype=bool)
    if len(points) == 0:
        return real
    hsv = rgb_to_hsv(points[:, 3:6])
    target = hue_of(rgb)
    dh = np.abs(hsv[:, 0] - target)
    dh = np.minimum(dh, 1.0 - dh)          # hue wraps at 1
    return (real & (dh <= hue_tol) & (hsv[:, 1] >= s_min)
            & (hsv[:, 2] >= v_min))


# ----------------------------------------------------------------------
# fixed-radius sphere fit
# ----------------------------------------------------------------------
def _fit_sphere_gn(xyz: np.ndarray, radius: float, c0: np.ndarray,
                   iters: int = 40) -> dict:
    """Robust Gauss-Newton fit of min sum rho(||p - c|| - R) with fixed R."""
    c = np.asarray(c0, dtype=np.float64).copy()
    scale = 0.0025                          # Huber band (m)
    for _ in range(iters):
        d = xyz - c
        r = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
        res = r - radius
        w = np.where(np.abs(res) <= scale, 1.0, scale / np.abs(res))
        J = -d / r[:, None]
        H = (J * w[:, None]).T @ J + 1e-8 * np.eye(3)
        g = (J * w[:, None]).T @ res
        try:
            step = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            break
        n = float(np.linalg.norm(step))
        if n > 0.02:                        # never jump more than 2 cm
            step = step * (0.02 / n)
            n = 0.02
        c = c + step
        if n < 1e-7:
            break
    d = xyz - c
    r = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
    res = r - radius
    inl = np.abs(res) <= 0.004
    if int(inl.sum()) >= 6:
        rms = float(np.sqrt(np.mean(res[inl] ** 2)))
    else:
        rms = float("inf")
    # robust cost: median absolute residual (not dragged by silhouette blends)
    cost = float(np.median(np.abs(res)))
    return {"centre": c, "rms": rms, "n_inliers": int(inl.sum()),
            "cost": cost}


def fit_sphere_known_radius(xyz: np.ndarray, radius: float,
                            cam_pos: Optional[np.ndarray] = None) -> Optional[dict]:
    """Robust sphere fit with a KNOWN radius.

    Several initial centres are tried along the camera ray.  The visible
    surface of a 7 cm ball seen from ~0.8 m is nearly a hemisphere, so the
    true centre (behind the observed surface centroid) is well constrained;
    the multi-init search also rejects the mirrored near-degenerate solution
    when the visible cap is small.
    """
    if len(xyz) < 12:
        return None
    centroid = xyz.mean(axis=0)
    if cam_pos is None:
        u = np.array([0.0, 0.0, 1.0])
    else:
        v = np.asarray(cam_pos, dtype=np.float64) - centroid
        nv = float(np.linalg.norm(v))
        u = v / nv if nv > 1e-9 else np.array([0.0, 0.0, 1.0])
    best = None
    for s in (-0.9, -0.6, -0.3, 0.0, 0.3, 0.6):
        fit = _fit_sphere_gn(xyz, radius, centroid + s * radius * u)
        if best is None or fit["cost"] < best["cost"]:
            best = fit
    if best is None or best["n_inliers"] < 10:
        return None
    best["centre"] = best["centre"].astype(np.float64)
    best["n_points"] = int(len(xyz))
    return best


def estimate_green_ball(points: np.ndarray, radius: float = BALL_RADIUS,
                        cam_pos: Optional[np.ndarray] = None,
                        min_points: int = 20) -> Optional[dict]:
    """Vision-only GREEN target ball estimate from a dense (M,6) cloud."""
    m = segment_color(points, BALL_RGB)
    xyz = points[m][:, :3]
    if len(xyz) < min_points:
        return None
    fit = fit_sphere_known_radius(xyz, radius, cam_pos=cam_pos)
    if fit is None:
        return None
    c = fit["centre"]
    if np.any(c < BALL_XYZ_MIN) or np.any(c > BALL_XYZ_MAX):
        return None
    if not (0.0025 <= c[2] <= 0.50):
        return None
    fit["n_color_points"] = int(m.sum())
    return fit


# ----------------------------------------------------------------------
# box opening (static fixture): centre + rim height
# ----------------------------------------------------------------------
def estimate_box(points: np.ndarray, min_points: int = 40) -> Optional[dict]:
    """Estimate the box opening centre and rim height from the brown walls.

    The box is a static, symmetric open-top container.  The top faces of the
    four walls form a square ring, so the midpoint of their x/y extents is the
    opening centre and the high percentile of their z is the rim plane.  A
    plausibility gate (extent and centre near the scene fixture) rejects
    partial/occluded fits; the caller falls back to the fixture constants.

    Brown/anti-aliased outliers on the table occur a few centimetres outside
    the fixture (the point-cloud frame has a small offset), so the known
    static footprint is used only as a generous spatial GATE; the centre and
    rim height themselves still come from the observed points, which cancels
    the cloud-vs-simulator offset.
    """
    m = segment_color(points, BOX_RGB)
    xyz = points[m][:, :3]
    # generous static-fixture gate: the box never moves in this task
    gate = ((xyz[:, 0] > 0.185) & (xyz[:, 0] < 0.405)
            & (xyz[:, 1] > 0.035) & (xyz[:, 1] < 0.265))
    xyz = xyz[gate]
    if len(xyz) < min_points:
        return None
    z = xyz[:, 2]
    z_rim = float(np.percentile(z, 98))
    if not (0.20 < z_rim < 0.34):
        return None
    top = xyz[z > z_rim - 0.005]
    if len(top) < 25:
        return None
    lo = np.percentile(top[:, :2], 1, axis=0)
    hi = np.percentile(top[:, :2], 99, axis=0)
    centre = 0.5 * (lo + hi)
    extent = hi - lo
    # the outer footprint is 0.20 x 0.20 m; allow occlusion but reject junk
    if np.any(extent < 0.14) or np.any(extent > 0.24):
        return None
    if np.linalg.norm(centre - BOX_CENTER) > 0.025:
        return None
    return {
        "centre_xy": centre.astype(np.float64),
        "rim_z": z_rim,
        "extent_xy": extent.astype(np.float64),
        "n_points": int(len(xyz)),
        "n_top_points": int(len(top)),
    }


def fixture_box() -> dict:
    """Scene-constant box estimate used as a fallback / debugging prior."""
    return {
        "centre_xy": BOX_CENTER.copy(),
        "rim_z": float(BOX_RIM_Z),
        "extent_xy": np.array([0.20, 0.20], dtype=np.float64),
        "n_points": 0,
        "n_top_points": 0,
        "source": "fixture",
    }


# ----------------------------------------------------------------------
# ghost (virtual target) point model
# ----------------------------------------------------------------------
def sample_sphere_shell(radius: float, n: int = 64,
                        rng: Optional[np.random.Generator] = None
                        ) -> np.ndarray:
    """``n`` points uniformly on a sphere of ``radius`` (local frame)."""
    rng = rng or np.random.default_rng(1)
    v = rng.normal(size=(n, 3))
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    return (v * radius).astype(np.float32)
