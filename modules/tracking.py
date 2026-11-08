"""Module 2: Object tracking across video frames (ByteTrack via Ultralytics)."""
import os
import time

import cv2

from modules import detection


def analyze_video(video_path, out_path, conf=0.3, frame_skip=2, max_frames=600,
                  max_width=640, progress_cb=None, model_name=None, imgsz=640,
                  classes_text=""):
    """Tracks objects in a video. Returns a dict with stats and sample frames."""
    model = detection.get_model(model_name, classes_text, fresh=True)  # fresh tracker state
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open the video file.")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    writer = None
    ids_per_label = {}
    samples = []
    processed = 0
    index = 0
    start = time.time()

    while processed < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        index += 1
        if (index - 1) % frame_skip != 0:
            continue

        h, w = frame.shape[:2]
        if w > max_width:
            frame = cv2.resize(frame, (max_width, int(h * max_width / w)))

        result = model.track(frame, conf=conf, imgsz=imgsz, persist=True,
                             tracker="bytetrack.yaml", verbose=False)[0]
        if result.boxes is not None and result.boxes.id is not None:
            for cls, tid in zip(result.boxes.cls.tolist(), result.boxes.id.tolist()):
                ids_per_label.setdefault(result.names[int(cls)], set()).add(int(tid))

        annotated = result.plot()
        if writer is None:
            fh, fw = annotated.shape[:2]
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                     max(src_fps / frame_skip, 1), (fw, fh))
        writer.write(annotated)

        if processed % 40 == 0 and len(samples) < 6:
            samples.append(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

        processed += 1
        if progress_cb:
            progress_cb(min(index / total, 1.0))

    cap.release()
    if writer:
        writer.release()

    elapsed = max(time.time() - start, 0.001)
    return {
        "frames_processed": processed,
        "seconds": round(elapsed, 1),
        "fps": round(processed / elapsed, 1),
        "unique_objects": {k: len(v) for k, v in ids_per_label.items()},
        "samples": samples,
        "output_exists": os.path.exists(out_path),
    }
