"""Compile a video stream into an EvidenceModel of physical observables.

The witness runs a person detector / pose estimator over the frames and reduces
the track to physical observables — person present, body orientation, vertical
position, motion, rapid descent — each with a confidence. It reports facts, not
judgments; the consumer composes them.

Two backends:
  - "geometry"  a real vision model (torchvision Faster R-CNN / Keypoint R-CNN,
                imported lazily so this module loads without torch installed).
  - "stub"      deterministic observables from a supplied dict — for tests and
                for wiring a governor before a camera is attached.

A fall is an EVENT over time. A single prone frame can defeat a detector that
was trained on upright people; the descent (bounding-box centre dropping while
the aspect widens) is the more robust and more honest signal, and it is tracked
here across the frames where the person is still detected.
"""

from __future__ import annotations

import hashlib

from erisml_compiler.ir.evidence import EvidenceModel, PhysicalObservable


def _observables_from_track(track: list[dict], img_h: int) -> list[PhysicalObservable]:
    """Reduce a per-frame person track to physical observables.

    Each track entry: {score, x0,y0,x1,y1} in pixels (y down). Missing frames
    (no detection) are simply absent. img_h normalizes vertical position."""
    seen = [t for t in track if t is not None]
    if not seen:
        return [
            PhysicalObservable(
                name="person_present", value=0.0, confidence=0.0, note="no detection in any frame"
            )
        ]

    scores = [t["score"] for t in seen]
    present_conf = max(scores)

    # aspect (w/h): > 1 means wider than tall -> body horizontal
    def aspect(t):
        return (t["x1"] - t["x0"]) / max(1.0, (t["y1"] - t["y0"]))

    def ycenter(t):
        return (t["y1"] + t["y0"]) / 2.0

    last = seen[-1]
    horiz = aspect(last)
    horiz_val = max(0.0, min(1.0, (horiz - 0.6) / 1.4))  # 0.6->0, 2.0->1
    # confidence of the orientation read = detector score on that frame
    # vertical position: y_center near image bottom -> on floor
    yv = ycenter(last) / max(1.0, img_h)
    low_val = max(0.0, min(1.0, (yv - 0.45) / 0.45))  # 0.45->0 (mid), 0.9->1 (bottom)

    # rapid descent: increase in y_center over the detected span (normalized per frame)
    descent = 0.0
    if len(seen) >= 2:
        dy = (ycenter(seen[-1]) - ycenter(seen[0])) / max(1.0, img_h)
        descent = max(0.0, min(1.0, dy / 0.25))  # a 25%-of-frame drop -> 1.0
    # aspect widening over the span (fall onset even if the prone frame is lost)
    widen = 0.0
    if len(seen) >= 2:
        widen = max(0.0, min(1.0, (aspect(seen[-1]) - aspect(seen[0])) / 0.8))

    # descent/onset confidence: how much of the clip we actually tracked
    track_frac = len(seen) / max(1, len(track))
    onset_conf = round(present_conf * track_frac, 4)

    # fall_transition: the person was upright EARLY and horizontal LATE -- a fall
    # event, not a static posture. This is what separates a collapse from lying
    # down on purpose (yoga, sleeping): both end horizontal, only a fall began
    # upright. Robust to the descent underreading when a fall is mostly sideways.
    fall_transition = 0.0
    trans_conf = 0.0
    if len(seen) >= 3:
        k = max(1, len(seen) // 3)
        early = sum(aspect(t) for t in seen[:k]) / k
        late = sum(aspect(t) for t in seen[-k:]) / k
        upright_early = max(0.0, min(1.0, (0.95 - early) / 0.45))  # early < ~0.95 => upright
        horizontal_late = max(0.0, min(1.0, (late - 1.10) / 0.6))  # late > ~1.10 => horizontal
        fall_transition = round(upright_early * horizontal_late, 4)
        trans_conf = round(present_conf * track_frac, 4)

    return [
        PhysicalObservable(
            name="person_present",
            value=1.0,
            confidence=round(present_conf, 4),
            note=f"detected in {len(seen)}/{len(track)} frames",
        ),
        PhysicalObservable(
            name="body_horizontal",
            value=round(horiz_val, 4),
            confidence=round(last["score"], 4),
            note=f"bbox aspect w/h={horiz:.2f} on last detected frame",
        ),
        PhysicalObservable(
            name="on_floor",
            value=round(low_val, 4),
            confidence=round(last["score"], 4),
            note=f"bbox y_center={yv:.2f} of frame height",
        ),
        PhysicalObservable(
            name="rapid_descent",
            value=round(max(descent, widen), 4),
            confidence=onset_conf,
            note="y_center drop + aspect widening over the tracked span",
        ),
        PhysicalObservable(
            name="fall_transition",
            value=fall_transition,
            confidence=trans_conf,
            note="upright early AND horizontal late -- a fall event, not a static posture",
        ),
    ]


def _geometry_track(frames, min_score: float = 0.5) -> tuple[list[dict], int, str]:
    """Run a torchvision person detector over frames -> per-frame best-person box.
    Imported lazily so the module and its stub backend load without torch."""
    import numpy as np  # noqa: F401
    import torch
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn_v2,
        FasterRCNN_ResNet50_FPN_V2_Weights,
    )

    w = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    net = fasterrcnn_resnet50_fpn_v2(weights=w).eval()
    tf = w.transforms()
    track: list[dict] = []
    img_h = 0
    for fr in frames:
        arr = fr[:, :, :3]
        img_h = arr.shape[0]
        x = tf(torch.from_numpy(arr).permute(2, 0, 1))
        with torch.no_grad():
            o = net([x])[0]
        best = None
        for s, lab, b in zip(o["scores"], o["labels"], o["boxes"]):
            if int(lab) == 1 and float(s) >= min_score:  # COCO person == 1
                bx = b.tolist()
                best = {"score": float(s), "x0": bx[0], "y0": bx[1], "x1": bx[2], "y1": bx[3]}
                break
        track.append(best)
    return track, img_h, f"torchvision:{w.__class__.__name__}"


def encode_video(
    source,
    backend: str = "geometry",
    evidence_id: str | None = None,
    stub_track: list[dict] | None = None,
    img_h: int = 512,
    min_score: float = 0.5,
) -> EvidenceModel:
    """Compile `source` (a video path, a list of frames, or -- for stub -- ignored)
    into a finalized EvidenceModel.

    backend="geometry": read frames (mp4 path via imageio, or an ndarray sequence)
    and run the detector. backend="stub": build observables from `stub_track`
    (list of per-frame {score,x0,y0,x1,y1} or None), no torch needed."""
    src_repr = source if isinstance(source, str) else f"<{len(source)} frames>"
    eid = evidence_id or hashlib.sha256(str(src_repr).encode()).hexdigest()[:16]

    if backend == "stub":
        track = stub_track or []
        obs = _observables_from_track(track, img_h)
        return EvidenceModel(
            evidence_id=eid,
            modality="vision",
            source=str(src_repr),
            n_frames=len(track),
            observables=obs,
            detector="stub",
            source_sha256=hashlib.sha256(str(src_repr).encode()).hexdigest(),
        ).finalize()

    # geometry backend: load frames
    if isinstance(source, str):
        import imageio.v2 as iio

        rdr = iio.get_reader(source)
        frames = [f for f in rdr]
        raw = open(source, "rb").read()
        src_sha = hashlib.sha256(raw).hexdigest()
    else:
        frames = list(source)
        src_sha = hashlib.sha256(b"".join(bytes(f) for f in frames)).hexdigest()

    track, ih, detector = _geometry_track(frames, min_score=min_score)
    obs = _observables_from_track(track, ih or img_h)
    return EvidenceModel(
        evidence_id=eid,
        modality="vision",
        source=str(src_repr),
        n_frames=len(frames),
        observables=obs,
        detector=detector,
        source_sha256=src_sha,
    ).finalize()


class VideoWitnessStream:
    """A rolling-window witness for a live robot camera.

    A robot does not get a finished clip; it gets a frame at a time. Push frames
    as they arrive; the witness runs the detector on each, keeps the last
    `window` per-frame boxes, and `evidence()` returns the EvidenceModel over that
    rolling window at any moment. So a fall is caught while it happens: the
    descent shows up in the window as soon as the person starts going down.

    The per-frame detector is pluggable. The default is the lazy torchvision
    geometry detector; tests pass their own `detect_fn(frame) -> box|None` so the
    stream runs without torch.
    """

    def __init__(
        self,
        window: int = 16,
        min_score: float = 0.5,
        detect_fn=None,
        source: str = "camera://robot",
        img_h: int = 512,
    ):
        self.window = window
        self.min_score = min_score
        self.source = source
        self.img_h = img_h
        self._track: list = []
        self._n = 0
        self._detector = "stream:custom" if detect_fn else "stream:pending"
        if detect_fn is not None:
            self._detect = detect_fn
        else:
            self._detect = self._make_torch_detector()

    def _make_torch_detector(self):
        state = {}

        def detect(frame):
            if "net" not in state:
                import torch
                from torchvision.models.detection import (
                    fasterrcnn_resnet50_fpn_v2,
                    FasterRCNN_ResNet50_FPN_V2_Weights,
                )

                w = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
                state["torch"] = torch
                state["net"] = fasterrcnn_resnet50_fpn_v2(weights=w).eval()
                state["tf"] = w.transforms()
                self._detector = f"stream:torchvision:{w.__class__.__name__}"
            torch = state["torch"]
            x = state["tf"](torch.from_numpy(frame[:, :, :3]).permute(2, 0, 1))
            with torch.no_grad():
                o = state["net"]([x])[0]
            for s, lab, b in zip(o["scores"], o["labels"], o["boxes"]):
                if int(lab) == 1 and float(s) >= self.min_score:
                    bx = b.tolist()
                    return {"score": float(s), "x0": bx[0], "y0": bx[1], "x1": bx[2], "y1": bx[3]}
            return None

        return detect

    def push(self, frame) -> None:
        """Feed one camera frame (H,W,3 ndarray)."""
        try:
            self.img_h = frame.shape[0]
        except Exception:
            pass
        self._track.append(self._detect(frame))
        self._n += 1
        if len(self._track) > self.window:
            self._track = self._track[-self.window :]

    def evidence(self) -> EvidenceModel:
        """The EvidenceModel over the current rolling window."""
        obs = _observables_from_track(self._track, self.img_h)
        return EvidenceModel(
            evidence_id=f"{self.source}@{self._n}",
            modality="vision",
            source=self.source,
            n_frames=len(self._track),
            observables=obs,
            detector=self._detector,
            source_sha256=hashlib.sha256(f"{self.source}@{self._n}".encode()).hexdigest(),
        ).finalize()
