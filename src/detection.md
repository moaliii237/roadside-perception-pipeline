# Object detection

Fixed roadside objects are detected with **RF-DETR** (a real-time DETR-style
transformer detector), fine-tuned on the target classes: fence, gate,
guardrail, traffic sign, post or bollard, tree trunk, lamp post, mailbox,
picnic table, fire hydrant, utility cabinet.

## Design choices

- **Recall-first operating point.** The detector runs at a low confidence
  threshold so few real objects are missed; precision is recovered later by
  the multi-view fusion step (an object must be seen consistently across
  several frames to become a map object), not by a high per-frame threshold.

- **Raw pixel coordinates.** No lens undistortion is applied. Boxes are in the
  original image frame, and the geolocation step
  ([`monocular_geolocation.py`](monocular_geolocation.py)) consumes them
  directly. The camera geometry is recovered from the drive rather than from a
  calibration file.

- **Bonnet mask.** The vehicle bonnet reflects the scene and produces false
  detections, so the same fixed mask is applied at training and at inference,
  keeping the input distribution identical.

- **Class gating.** A class the model is not yet good enough at is held back
  from the published output rather than shipping low-quality detections; it
  returns to the map after a retraining round.

## Improvement loop

Detections that a human reviewer rejects (for example storage cages mistaken
for fences) are fed back as hard negatives in the next training round. A
segmentation-assist model (e.g. SAM) can speed up labeling the new data by
turning a click or a box into a full mask.
