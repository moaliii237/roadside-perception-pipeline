"""Monocular geolocation of detected objects from a moving camera.

Given a GPS track and per-frame object detections (bounding boxes in raw
pixel coordinates), estimate a real-world position for each object by fusing
the bearing lines from multiple vehicle positions. A single frame cannot
give range; motion does.

This is a clean, self-contained reference implementation of the method. It
uses only numpy and takes generic inputs, so it can be read and run without
any particular dataset.

Pipeline:
    track (lat, lon, t per frame)  ->  local ENU + per-frame heading
    detections (frame, bbox)       ->  bearing (+ optional ground-plane range)
    per object                     ->  robust multi-view least-squares solve

Author: Moali Jaberi.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

M_PER_DEG = 111_320.0


# --------------------------------------------------------------------------- #
#  Camera model (pinhole, no distortion). Values are examples; the pitch can
#  be recovered from the footage with calibrate_pitch_from_horizon() below.
# --------------------------------------------------------------------------- #
@dataclass
class Camera:
    fx: float = 1111.0          # focal length in pixels
    img_w: int = 1280
    img_h: int = 720
    height_m: float = 1.5       # camera height above the road
    pitch_rad: float = 0.0      # downward pitch; 0 = optical axis horizontal
    yaw_rad: float = 0.0        # mount yaw offset

    @property
    def cx(self) -> float:
        return self.img_w / 2.0

    @property
    def horizon_row(self) -> float:
        """Image row of the horizon for the current pitch."""
        return self.img_h / 2.0 + self.fx * math.tan(self.pitch_rad)


# --------------------------------------------------------------------------- #
#  Track: GPS -> local ENU metres -> per-frame heading
# --------------------------------------------------------------------------- #
@dataclass
class Frame:
    idx: int
    t: float
    e: float
    n: float
    heading: float | None = None   # radians, compass (E component, N component)


def build_track(latlon_t: list[tuple[float, float, float]],
                chord_m: float = 5.0, min_chord_m: float = 3.0) -> list[Frame]:
    """Convert (lat, lon, t) samples to ENU frames with a smoothed heading.

    Heading uses a distance-gated chord so that slow/stationary stretches do
    not produce noise-driven bearings, then a short circular smoothing.
    """
    lat0 = float(np.mean([p[0] for p in latlon_t]))
    lon0 = float(np.mean([p[1] for p in latlon_t]))
    coslat = math.cos(math.radians(lat0))
    frames = [
        Frame(i, t,
              (lon - lon0) * M_PER_DEG * coslat,
              (lat - lat0) * M_PER_DEG)
        for i, (lat, lon, t) in enumerate(latlon_t)
    ]
    s = np.concatenate([[0.0], np.cumsum([
        math.hypot(frames[i].e - frames[i - 1].e, frames[i].n - frames[i - 1].n)
        for i in range(1, len(frames))])])
    raw = np.full(len(frames), np.nan)
    for i in range(len(frames)):
        j0 = int(np.searchsorted(s, s[i] - chord_m))
        j1 = int(np.searchsorted(s, s[i] + chord_m)) - 1
        j0, j1 = min(j0, i), min(max(j1, i), len(frames) - 1)
        de, dn = frames[j1].e - frames[j0].e, frames[j1].n - frames[j0].n
        if math.hypot(de, dn) >= min_chord_m:
            raw[i] = math.atan2(de, dn)          # compass bearing
    # circular smoothing (window 7) with forward/back fill of gaps
    k = np.ones(7)
    sin_s = np.convolve(np.nan_to_num(np.sin(raw)), k, "same")
    cos_s = np.convolve(np.nan_to_num(np.cos(raw)), k, "same")
    valid = np.convolve((~np.isnan(raw)).astype(float), k, "same")
    with np.errstate(invalid="ignore"):
        head = np.where(valid > 0, np.arctan2(sin_s, cos_s), np.nan)
    last = np.nan
    for i in range(len(frames)):
        if not math.isnan(head[i]):
            last = head[i]
        frames[i].heading = None if math.isnan(last) else float(last)
    return frames, (lat0, lon0, coslat)


def enu_to_latlon(e, n, origin):
    lat0, lon0, coslat = origin
    return (lat0 + n / M_PER_DEG, lon0 + e / (M_PER_DEG * coslat))


# --------------------------------------------------------------------------- #
#  Horizon self-calibration (recover pitch from road-edge vanishing points)
# --------------------------------------------------------------------------- #
def pitch_from_horizon_row(cam: Camera, horizon_row_px: float) -> float:
    """Invert Camera.horizon_row: given the measured horizon row, get pitch."""
    return math.atan2(horizon_row_px - cam.img_h / 2.0, cam.fx)


# --------------------------------------------------------------------------- #
#  Per-detection projection: bearing (always) and range (when trustworthy)
# --------------------------------------------------------------------------- #
@dataclass
class Detection:
    frame_idx: int
    bbox: tuple[float, float, float, float]   # x1, y1, x2, y2 (raw pixels)
    bearing: float | None = None
    range_m: float | None = None
    flags: list[str] = field(default_factory=list)


def project(det: Detection, frame: Frame, cam: Camera,
            min_px_below_horizon: float = 15.0, max_range_m: float = 25.0) -> None:
    """Fill det.bearing and (when the geometry is trustworthy) det.range_m."""
    if frame.heading is None:
        det.flags.append("no_heading")
        return
    x1, y1, x2, y2 = det.bbox
    u = (x1 + x2) / 2.0
    v_bottom = y2
    alpha = math.atan2(u - cam.cx, cam.fx)         # angle off the optical axis
    det.bearing = frame.heading + alpha + cam.yaw_rad
    below = v_bottom - cam.horizon_row
    if below < min_px_below_horizon:
        det.flags.append("near_horizon")           # too flat to range reliably
        return
    r = (cam.fx * cam.height_m) / below            # ground-plane range
    if r > max_range_m:
        det.flags.append("too_far")
        return
    det.range_m = r


# --------------------------------------------------------------------------- #
#  Multi-view fusion: robust least squares over one object's sightings
# --------------------------------------------------------------------------- #
@dataclass
class ObjectFix:
    e: float
    n: float
    n_sightings: int
    bearing_spread_deg: float
    quality: str


def fuse(sightings: list[tuple[Detection, Frame]],
         sigma_theta_deg: float = 1.5, range_frac: float = 0.30,
         iters: int = 4) -> ObjectFix | None:
    """Solve for one object position from many bearings (+ near ranges).

    Each bearing contributes a line; near-range detections also contribute a
    distance constraint. A Huber reweighting damps outlier sightings.
    """
    obs = [(d, f) for d, f in sightings if d.bearing is not None]
    if not obs:
        return None
    sigma_theta = math.radians(sigma_theta_deg)
    w = {id(d): 1.0 for d, _ in obs}
    x = None
    for _ in range(iters):
        A = np.zeros((2, 2))
        b = np.zeros(2)
        for d, f in obs:
            p = np.array([f.e, f.n])
            th = d.bearing
            dvec = np.array([math.sin(th), math.cos(th)])     # along bearing
            nvec = np.array([math.cos(th), -math.sin(th)])    # perpendicular
            r_ref = min(max(d.range_m or 25.0, 5.0), 60.0)
            wb = w[id(d)] / (r_ref * sigma_theta) ** 2
            A += wb * np.outer(nvec, nvec)
            b += wb * nvec * (nvec @ p)
            if d.range_m is not None:
                wr = 1.0 / (range_frac * d.range_m) ** 2
                A += wr * np.outer(dvec, dvec)
                b += wr * dvec * (dvec @ p + d.range_m)
        if np.linalg.det(A) < 1e-9:
            return None
        x = np.linalg.solve(A, b)
        for d, f in obs:                                       # Huber reweight
            p = np.array([f.e, f.n])
            nvec = np.array([math.cos(d.bearing), -math.sin(d.bearing)])
            r_ref = min(max(d.range_m or 25.0, 5.0), 60.0)
            res = abs(nvec @ (x - p)) / (r_ref * sigma_theta)
            w[id(d)] = 1.0 if res <= 1.345 else 1.345 / res
    bearings = [d.bearing for d, _ in obs]
    spread = math.degrees(max(bearings) - min(bearings)) if len(obs) > 1 else 0.0
    n_near = sum(1 for d, _ in obs if d.range_m is not None)
    if spread >= 5.0 and len(obs) >= 5:
        quality = "triangulated"
    elif n_near >= 2:
        quality = "near_range"
    else:
        quality = "weak"
    return ObjectFix(float(x[0]), float(x[1]), len(obs), round(spread, 1), quality)


# --------------------------------------------------------------------------- #
#  Tiny synthetic demo: a vehicle drives north past one object 8 m to the east
# --------------------------------------------------------------------------- #
def _demo() -> None:
    cam = Camera(pitch_rad=math.radians(3.0))
    # straight northward drive, 1 m spacing, at a fixed origin
    lat0, lon0 = 52.0, 5.0
    coslat = math.cos(math.radians(lat0))
    latlon_t = [(lat0 + (i * 1.0) / M_PER_DEG, lon0, float(i)) for i in range(20)]
    frames, origin = build_track(latlon_t)

    obj_e, obj_n = 8.0, 12.0          # true object position in ENU metres
    sightings = []
    for f in frames:
        if f.heading is None:
            continue
        de, dn = obj_e - f.e, obj_n - f.n
        rng = math.hypot(de, dn)
        if rng > 25:
            continue
        bearing_true = math.atan2(de, dn)
        alpha = bearing_true - f.heading
        u = cam.cx + cam.fx * math.tan(alpha)
        if not (0 <= u <= cam.img_w):
            continue
        below = cam.fx * cam.height_m / rng
        v_bottom = cam.horizon_row + below
        det = Detection(f.idx, (u - 10, v_bottom - 30, u + 10, v_bottom))
        project(det, f, cam)
        sightings.append((det, f))

    fix = fuse(sightings)
    est = enu_to_latlon(fix.e, fix.n, origin)
    true = enu_to_latlon(obj_e, obj_n, origin)
    err = math.hypot(fix.e - obj_e, fix.n - obj_n)
    print(f"sightings used : {fix.n_sightings}")
    print(f"quality        : {fix.quality} (spread {fix.bearing_spread_deg} deg)")
    print(f"estimated      : {est[0]:.6f}, {est[1]:.6f}")
    print(f"true           : {true[0]:.6f}, {true[1]:.6f}")
    print(f"error          : {err:.2f} m")


if __name__ == "__main__":
    _demo()
