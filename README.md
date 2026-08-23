# Roadside Perception Pipeline

A computer-vision pipeline that turns a forward-facing car camera (plus GPS)
into a map of roadside objects and vegetation. Built for a roadside-inspection
use case: a vehicle drives the roads, and the system produces, per object, a
class, a confidence, an evidence crop, and a real map position.

This repository documents the methods I designed and implemented. It contains
clean, generalized reference code, not any client data, credentials, or
proprietary application code.

## What the pipeline does

```
raw driving footage (RGB frames + GPS track)
        │
        ├─ object detection ........ fixed roadside objects (fences, gates,
        │                            guardrails, signs, posts, tree trunks,
        │                            lamp posts, cabinets, ...)
        │
        ├─ vegetation segmentation .. short mown grass / tall grass /
        │                            woody scrub / other
        │
        ├─ monocular geolocation .... each detection gets its own map
        │                            position from camera geometry + the
        │                            GPS track (not the vehicle position)
        │
        └─ per-object fusion ........ many sightings of one object are
                                     merged into one confident map object
```

## The interesting part: monocular geolocation

A single camera cannot measure distance from one frame. The trick is motion:
the same object seen from several vehicle positions gives several bearing
lines that intersect where the object actually stands. The method in
[`src/monocular_geolocation.py`](src/monocular_geolocation.py):

1. **Per-frame heading** from the GPS track (distance-gated chord, circular
   smoothing) so short stops and GPS jitter do not corrupt the bearing.
2. **Horizon self-calibration** from the footage itself (vanishing point of
   the road edges) to recover the camera pitch, instead of trusting a spec
   sheet.
3. **Ground-plane range** for each detection from its pixel height below the
   horizon, with an explicit trust gate (too close to the horizon or beyond a
   max range → bearing-only).
4. **Multi-view fusion**: a robust (Huber) least-squares solve over all
   bearings and near-range measurements of a tracked object → one position
   with a covariance and a quality label (triangulated / near-range / weak).
5. **Validation** against aerial imagery: the produced positions were checked
   to land within a few metres of the real objects.

No lens undistortion is assumed; all detections are in raw pixel coordinates,
and the calibration is recovered from the drive.

## Semantic components

- **Object detection** — a transformer detector (RF-DETR) fine-tuned for the
  fixed-object classes, run with a recall-first operating point and filtered
  downstream. See [`src/detection.md`](src/detection.md).
- **Vegetation segmentation** — a frozen DINOv2 ViT-S/14 backbone with a small
  trained convolutional head, 4 vegetation classes. Architecture in
  [`src/vegetation_segmentation.py`](src/vegetation_segmentation.py). Trees are
  intentionally handled by the object detector (as trunks), not the vegetation
  model, because a countable trunk is more useful on a map than a painted crown.

## Data engineering

The pipeline was fed from ROS 2 recordings (`.mcap`, camera + LiDAR + GPS +
IMU + thermal topics). Handling included: reading topics with `mcap` /
`mcap-ros2-support`, time-interpolating the 5 Hz GPS onto each camera frame,
and a full integrity audit of a large recording set (open every archive,
read every recording, compare real per-topic message counts against the
metadata) before processing.

## Skills demonstrated

Computer vision (detection fine-tuning, semantic segmentation with
self-supervised backbones), classical multi-view geometry, sensor fusion,
ROS 2 / mcap data handling, large-dataset auditing, and packaging results as
GeoJSON for a mapping frontend.

## License

MIT, see [LICENSE](LICENSE). The reference code here is my own, written to
illustrate the methods. It ships no trained weights and no third-party data.
