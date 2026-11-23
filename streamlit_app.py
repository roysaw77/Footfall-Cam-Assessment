"""Streamlit app — staff (name-tag) detection over a video, using the hosted Roboflow workflow.

Run:
    .venv\\Scripts\\streamlit run streamlit_app.py

Pick a video, press *Generate video*, and the app produces:
  * an annotated mp4 — one Staff box + one Tag box per frame, each labelled with its
    (x, y) centre, plus a header line with the staff coordinate and timecode
  * a CSV with one row per frame (frame, time, staff conf/xy, tag conf/xy, staff_present)

The UI has exactly two controls: the video uploader and the *Generate video* button.
Everything else is fixed in the constants below (CONF, STRIDE, WORKERS, GAP_SEC).

Inference runs on Roboflow's serverless API (workspace `ge-rui`, workflow
`staff-tag-detection-gfsse`, model RF-DETR NAS), not on a local checkpoint. That means:
  * no ultralytics / torch / GPU on this machine,
  * every frame sent is one billed serverless image,
  * the round trip is ~0.7 s per frame, so calls are parallelised (WORKERS).

The API key is read from `st.secrets` when that is configured (Streamlit Cloud, or a
local `.streamlit/secrets.toml`), falling back to `roboflow_api` in `.env`.
"""

import csv
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
from inference_sdk import InferenceHTTPClient

SERVERLESS_URL = "https://serverless.roboflow.com"
WS = "ge-rui"
WF = "staff-tag-detection-gfsse"
STAFF_CLASS = "staff"
TAG_CLASS = "staff_tag"

# Run settings — fixed, not exposed in the UI.
CONF = 0.40        # confidence sent to the workflow; boxes below it never come back
STRIDE = 1         # 1 = every frame, so the output video has no skipped frames
WORKERS = 10       # concurrent API calls (~0.7 s per frame round trip)
GAP_SEC = 1.5      # merge detections closer than this into one presence segment

OUT_ROOT = Path("runs/app")
ENV_PATH = Path(".env")

st.set_page_config(page_title="FootfallCam — staff tag detection", page_icon="🎯", layout="wide")


# --------------------------------------------------------------------------- API key
def secret_api_key() -> str:
    """Read the key from st.secrets (Streamlit Cloud, or .streamlit/secrets.toml).

    Accepts either a top-level entry or a [roboflow] table:

        roboflow_api = "..."          # top level
        [roboflow]                    # or
        api_key = "..."
    """
    try:
        secrets = st.secrets
    except Exception:  # no secrets file configured at all
        return ""
    for name in ("roboflow_api", "roboflow_api_key", "ROBOFLOW_API_KEY"):
        try:
            if name in secrets:
                value = str(secrets[name]).strip()
                if value:
                    return value
        except Exception:
            continue
    try:
        if "roboflow" in secrets:
            value = str(secrets["roboflow"].get("api_key", "")).strip()
            if value:
                return value
    except Exception:
        pass
    return ""


def env_api_key() -> str:
    """Read roboflow_api from .env, if present."""
    try:
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("roboflow_api"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return ""


def load_api_key() -> str:
    """st.secrets wins when present, otherwise fall back to .env."""
    return secret_api_key() or env_api_key()


@st.cache_resource(show_spinner=False)
def make_client(key: str) -> InferenceHTTPClient:
    return InferenceHTTPClient(api_url=SERVERLESS_URL, api_key=key)


def to_xyxy(p: dict) -> list[float]:
    """Roboflow returns centre-based x, y, w, h."""
    cx, cy, w, h = p["x"], p["y"], p["width"], p["height"]
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def centre(box):
    x1, y1, x2, y2 = box
    return int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))


def draw_label(frame, box, text, colour):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    ty = max(th + 6, y1 - 4)
    cv2.rectangle(frame, (x1, ty - th - 6), (x1 + tw + 8, ty), colour, -1)
    cv2.putText(frame, text, (x1 + 4, ty - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                cv2.LINE_AA)


def draw_header(frame, lines):
    pad, lh = 10, 22
    width = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0] for t in lines) + 2 * pad
    roi = frame[0:lh * len(lines) + pad, 0:width]
    roi[:] = (roi * 0.35).astype(roi.dtype)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (pad, pad + lh * (i + 1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)


def infer(client, frame_bgr, conf: float, retries: int = 3) -> dict:
    """One workflow call for one frame. Returns raw staff / staff_tag detections."""
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        return {"staff": [], "staff_tag": [], "n": 0, "error": "jpeg encode failed"}
    import base64

    payload = base64.b64encode(buf.tobytes()).decode()
    delay, last = 1.0, None
    for _ in range(retries):
        try:
            res = client.run_workflow(workspace_name=WS, workflow_id=WF,
                                      images={"image": payload},
                                      parameters={"confidence": conf})
            preds = (res[0].get("predictions") or {}).get("predictions") or []
            return {"staff": [p for p in preds if p.get("class") == STAFF_CLASS],
                    "staff_tag": [p for p in preds if p.get("class") == TAG_CLASS],
                    "n": len(preds)}
        except Exception as exc:  # noqa: BLE001 - retry transient network/rate errors
            last = exc
            time.sleep(delay)
            delay = min(delay * 2, 10)
    return {"staff": [], "staff_tag": [], "n": 0, "error": str(last)}


def process(source: Path, client, conf: float, stride: int, workers: int, progress=None):
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError("could not open the video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()
    if not w or not h:
        raise RuntimeError("could not read the video dimensions")

    indices = list(range(0, n_total, max(stride, 1)))
    results: dict[int, dict] = {}
    t0 = time.time()
    batch = max(workers * 4, 8)

    # ---- pass 1: inference, in bounded batches so memory stays flat
    for start in range(0, len(indices), batch):
        chunk = indices[start:start + batch]
        cap = cv2.VideoCapture(str(source))
        frames = []
        for idx in chunk:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                frames.append((idx, frame))
        cap.release()

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(infer, client, fr, conf): idx for idx, fr in frames}
            for fut, idx in futs.items():
                results[idx] = fut.result()

        done = min(start + len(chunk), len(indices))
        if progress is not None:
            progress(done / max(len(indices), 1), done, len(indices))

    # ---- pass 2: render + rows
    run_dir = OUT_ROOT / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    video_path = run_dir / f"{source.stem}_tagged.mp4"
    csv_path = run_dir / f"{source.stem}_detections.csv"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    rows, idx = [], 0
    cap = cv2.VideoCapture(str(source))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        canvas = frame.copy()
        if idx in results:
            r = results[idx]
            staff = max(r["staff"], key=lambda p: p.get("confidence", 0)) if r["staff"] else None
            tag = max(r["staff_tag"], key=lambda p: p.get("confidence", 0)) if r["staff_tag"] else None
            sbox = to_xyxy(staff) if staff else None
            tbox = to_xyxy(tag) if tag else None
            sxy = centre(sbox) if sbox else None
            txy = centre(tbox) if tbox else None
            t_sec = idx / fps

            rows.append({
                "frame": idx,
                "time_sec": round(t_sec, 3),
                "staff_present": 1 if staff else 0,
                "staff_conf": round(staff.get("confidence", 0.0), 4) if staff else 0.0,
                "staff_x": sxy[0] if sxy else None,
                "staff_y": sxy[1] if sxy else None,
                "staff_bbox": " ".join(f"{v:.1f}" for v in sbox) if sbox else "",
                "tag_conf": round(tag.get("confidence", 0.0), 4) if tag else 0.0,
                "tag_x": txy[0] if txy else None,
                "tag_y": txy[1] if txy else None,
            })
            if sbox:
                draw_label(canvas, sbox, "Staff (%d,%d) %.2f" % (sxy[0], sxy[1], staff["confidence"]),
                           (255, 120, 0))
            if tbox:
                draw_label(canvas, tbox, "Tag (%d,%d) %.2f" % (txy[0], txy[1], tag["confidence"]),
                           (0, 0, 255))
            head = ["t=%6.2fs  frame %d/%d" % (t_sec, idx, n_total),
                    ("staff xy=(%d, %d)" % sxy) if sxy else "staff: not detected"]
            draw_header(canvas, head)
        # frames skipped by `stride` are passed through untouched: no boxes, no overlay
        writer.write(canvas)
        idx += 1

    cap.release()
    writer.release()

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("no frames were processed")
    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)

    return {
        "video": video_path,
        "csv": csv_path,
        "df": df,
        "fps": fps,
        "frames": len(df),
        "frames_total": n_total,
        "size": (w, h),
        "images_sent": len(indices),
        "elapsed": time.time() - t0,
        "conf": conf,
        "stride": stride,
    }


def segments(df: pd.DataFrame, gap_sec: float = 1.5):
    """Merge detections closer than `gap_sec` into presence segments."""
    if df.empty or not df["staff_present"].any():
        return []
    hit = df.index[df["staff_present"] == 1].tolist()
    dt = float(df["time_sec"].iloc[1] - df["time_sec"].iloc[0]) if len(df) > 1 else 0.04
    gap_frames = max(1, int(round(gap_sec / max(dt, 1e-6))))
    out, start, prev = [], hit[0], hit[0]
    for i in hit[1:]:
        if i - prev > gap_frames:
            out.append((start, prev))
            start = i
        prev = i
    out.append((start, prev))
    return [(df["time_sec"][a], df["time_sec"][b], b - a + 1, df["staff_conf"][a:b + 1].max())
            for a, b in out]


# --------------------------------------------------------------------------- UI
def main():
    st.title("🎯 Staff tag detection — video → annotated mp4 + CSV")
    st.caption(
        f"Hosted Roboflow workflow `{WS}/{WF}` (RF-DETR NAS, classes: `{STAFF_CLASS}`, "
        f"`{TAG_CLASS}`). Runs on Roboflow's serverless API — no local model, no GPU here."
    )

    with st.sidebar:
        uploaded = st.file_uploader("Video (mp4 / mov / avi / mkv)", type=["mp4", "mov", "avi", "mkv"])
        go = st.button("Generate video", type="primary", width="stretch")

    source = None
    if uploaded is not None:
        up_dir = OUT_ROOT / "uploads"
        up_dir.mkdir(parents=True, exist_ok=True)
        source = up_dir / uploaded.name
        source.write_bytes(uploaded.getbuffer())

    if source is None:
        st.info("Upload a video to begin.")
        st.stop()

    api_key = load_api_key()
    if not api_key:
        st.error("No Roboflow API key found. Add `roboflow_api` to `.streamlit/secrets.toml` "
                 "(or the app's Secrets in Streamlit Cloud), or to `.env` next to this app.")
        st.stop()

    left, right = st.columns([3, 2])
    with left:
        st.video(str(source))
    with right:
        cap = cv2.VideoCapture(str(source))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        cap.release()
        st.metric("frames", f"{n:,}")
        st.metric("fps", f"{fps:.2f}")
        st.metric("duration", f"{n / fps:.1f} s" if fps else "-")

    if go:
        bar = st.progress(0.0, text="starting…")

        def tick(frac, done, total):
            bar.progress(min(frac, 1.0), text=f"frame {done}/{total}")

        try:
            client = make_client(api_key)
            with st.spinner("running the hosted workflow…"):
                result = process(source, client, CONF, STRIDE, WORKERS, tick)
        except Exception as exc:                     # noqa: BLE001 - surface it in the UI
            st.error(f"failed: {exc}")
            st.stop()
        bar.progress(1.0, text="done")
        st.session_state["result"] = result

    result = st.session_state.get("result")
    if not result:
        st.stop()

    df = result["df"]
    segs = segments(df, GAP_SEC)
    staff_frames = int(df["staff_present"].sum())

    st.divider()
    st.subheader("Result")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("frames processed", f"{result['frames']:,}")
    c2.metric("staff frames", f"{staff_frames:,}",
              f"{100 * staff_frames / max(result['frames'], 1):.1f}%")
    c3.metric("segments", len(segs))
    c4.metric("images sent", f"{result['images_sent']:,}")
    st.caption(f"{result['frames_total']:,} source frames · confidence {result['conf']:.2f} · "
               f"stride {result['stride']} · {result['elapsed']:.0f}s")

    st.video(str(result["video"]))

    left, right = st.columns(2)
    left.download_button("⬇ annotated mp4", result["video"].read_bytes(),
                         file_name=result["video"].name, mime="video/mp4", width="stretch")
    right.download_button("⬇ detections csv", result["csv"].read_bytes(),
                          file_name=result["csv"].name, mime="text/csv", width="stretch")

    if segs:
        st.markdown("**Presence segments** (detections merged across gaps ≤ %.1fs)" % GAP_SEC)
        st.dataframe(pd.DataFrame(segs, columns=["start (s)", "end (s)", "frames", "max conf"]).round(3),
                     width="stretch", hide_index=True)
    else:
        st.warning("no frame had a staff detection — the workflow's confidence threshold may be too high")

    st.markdown("**Staff confidence over time**")
    st.line_chart(df.set_index("time_sec")["staff_conf"], height=220)

    with st.expander("per-frame table"):
        st.dataframe(df, width="stretch", hide_index=True, height=320)

    st.caption(f"files written to `{result['video'].parent}`")


if __name__ == "__main__":
    main()
