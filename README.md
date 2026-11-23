# Staff Identification from FootfallCam Footage

Solution documentation for the FootfallCam AI evaluation: identify every frame in which a
staff member appears (Task 1), and report that person's `(x, y)` position in each of those
frames (Task 2).

The solution is a small Streamlit application backed by a trained object-detection model
hosted on Roboflow. There is no local model, no GPU requirement, and no hand-written colour or
geometry heuristic anywhere in the code.

---

## 1. How staff are identified

The brief defines staff by a single visible feature: a rectangular name tag with dark
horizontal markings worn on the chest. Everything about the approach follows from that.

Rather than build a tag detector and then reason about who owns the tag, the model was trained
with **two classes on the same image**: `staff` (the person) and `staff_tag` (the badge). One
forward pass therefore returns both the body and the badge, already associated, with no
tracking, matching, or distance heuristic needed to join them.

The tag is a small target — roughly 20–25 pixels wide in 960×720 footage — which is why the
detector sees the whole frame, resized to the model's 640×640 input, rather than a cropped
region around a guess at where the chest is.

## 2. Pipeline

```
video file
   │
   ├─ decode frame (OpenCV), JPEG-encode at quality 92, base64
   │
   ├─ POST → Roboflow serverless workflow ................ 10 concurrent calls
   │      ge-rui / staff-tag-detection-gfsse
   │      RF-DETR NAS, classes: staff, staff_tag
   │      parameter: confidence = 0.40
   │
   ├─ JSON predictions → per-frame classification
   │      staff_present = any "staff" box
   │      (x, y) = centre of the highest-confidence "staff" box
   │
   └─ two outputs: annotated MP4  +  per-frame CSV
```

Inference is stateless per frame, so frames are independent and can be dispatched to a thread
pool without ordering constraints. The app runs in two passes: pass one performs all inference
and keeps only the predictions (a few hundred bytes per frame); pass two re-reads the video and
renders. Holding predictions rather than decoded frames keeps memory flat regardless of clip
length.

Calls are batched at `workers × 4` frames with three retries and exponential backoff (1 s → 10 s)
so a rate-limit response degrades throughput instead of dropping frames.

## 3. Model training

The detector is RF-DETR NAS, trained on Roboflow from the `staff-tag` annotation group in this
workspace, with `staff` and `staff_tag` as the two classes.

| Setting | Value |
|---|---|
| Architecture | RF-DETR NAS (`rfdetr-nas-pecoret-t1`) |
| Dataset | 1,329 images — 930 train / 200 validation / 199 test |
| Model input | 640×640 |
| Version | v2, Auto-Orient applied |
| Training preprocessing | none beyond auto-orient |
| Augmentation | horizontal flip, probability 0.5, train split only |

![Training recipe](docs/images/training-recipe.png)

Only horizontal flip was applied, and only to the train split. Horizontal mirroring is safe
here because a person facing left and a person facing right are both valid observations;
vertical flip is not, since an upside-down person never appears under a ceiling-mounted sensor.
With no other preprocessing, the images reach the model at their original geometry and are
resized to 640×640 inside the model.

![RF-DETR training metrics](docs/images/training-metrics.png)

Training converged quickly. mAP rose from about 61 % to about 91 % over 16 epochs, and
mAP@50:95 from about 30 % to about 51 %. Most of the gain arrives in the first three epochs;
after epoch 8 the curve oscillates in a band around 87–92 % with no sustained improvement.
That plateau is the useful signal — the model is limited by the dataset rather than by training
time, so adding epochs would not help and labelling more edge cases would.

## 4. Outputs

**Annotated MP4** — one orange `Staff` box and one red `Tag` box per frame, each labelled with
its centre coordinate and confidence, plus a header bar carrying the timecode and the staff
`(x, y)`.

**CSV**, one row per frame:

| Column | Meaning |
|---|---|
| `frame`, `time_sec` | source frame index and timestamp |
| `staff_present` | `1` when a `staff` detection passed the threshold |
| `staff_conf`, `staff_x`, `staff_y`, `staff_bbox` | Task 2 — confidence, centre point, and box |
| `tag_conf`, `tag_x`, `tag_y` | badge detection, where visible |

**Task 1** is the set of rows with `staff_present = 1`; the app also merges hits separated by
less than 1.5 s into presence segments and lists them with start/end times and peak confidence.

## 5. Results

A full pass over the bundled 53.6-second clip (960×720, 25 fps, 1,341 frames) at the app's
settings:

| Metric | Value |
|---|---|
| Frames processed | 1,341 of 1,341 |
| Frames with staff present | 299 (22.3 %) |
| Presence segments | 2 — 13.32–22.36 s and 32.24–49.72 s |
| Median staff confidence | 0.879 (max 0.921) |
| Frames with staff **and** tag | 280 of 299 |
| Wall-clock time | 231 s (≈ 6 frames/s) |

Confidence separated cleanly: 265 of the 299 detections scored 0.80 or above, and only five
scored below 0.60. Detections appear at the top of frame near t = 15 s and at the bottom
centre around t = 36 s, consistent with staff crossing the sales floor.

## 6. Why this generalises to unseen videos

The evaluation is scored on unseen footage, on the spot, so nothing in the pipeline is tied to
this particular clip:

- **No colour heuristics.** The old "bright white rectangle" idea breaks against white shirts
  and poor lighting. The detector learned the tag's appearance from data instead (section 3).
- **No hardcoded frame ranges or camera geometry.** Frames are processed individually; there
  is no assumption about where in the frame a person appears or when.
- **Occlusion tolerance comes from the two-class design.** When someone turns away and the tag
  is hidden, `staff_tag` disappears but `staff` still fires, so frame-level presence survives
  the exact moment a tag-only detector would fail.
- **Resolution independence.** The model returns coordinates in source pixels, so the same
  model works on other sensor heights and frame sizes.

## 7. Constraints and next steps

- **Network dependency.** Inference is remote, so the app needs connectivity and an API key in
  `.env`. The NAS model's weights cannot be downloaded, so there is no offline fallback. A
  local YOLO checkpoint would be the safer option for a live demo.
- **No temporal tracking.** Frames are classified independently, so a single missed detection
  produces a one-frame flicker and there is no Kalman smoothing on the trajectory yet. The CSV
  already carries everything needed to add it.
- **Cost.** Every frame is one billed serverless image; a full pass costs on the order of a
  quarter of a credit at published RF-DETR rates.

## 8. Running it

```bash
pip install -r requirements.txt      # streamlit, opencv, inference-sdk
# .env must contain: roboflow_api=<your key>
.venv\Scripts\streamlit run streamlit_app.py
```

The interface has two controls — upload a video, press **Generate video** — and produces the
annotated MP4 and CSV for download. Run settings are constants at the top of
`streamlit_app.py`:

| Constant | Value | Purpose |
|---|---|---|
| `CONF` | 0.40 | confidence threshold sent to the workflow |
| `STRIDE` | 1 | process every frame |
| `WORKERS` | 10 | concurrent inference calls |
| `GAP_SEC` | 1.5 | merge gap when forming presence segments |
