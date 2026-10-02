"""EvidenceModel + video-witness (stub backend, no torch)."""

import hashlib
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from erisml_compiler.ingestion import VideoWitnessStream, encode_video
from erisml_compiler.ir import (
    EvidenceModel,
    PhysicalObservable,
    SensorAttestation,
    check_attestation,
)


def _upright_then_fall_track(img_h=512):
    """A person tracked upright, then descending as they fall (y_center rising,
    aspect widening), then lost once fully prone."""
    return [
        {"score": 0.98, "x0": 200, "y0": 90, "x1": 280, "y1": 300},   # upright, tall
        {"score": 0.95, "x0": 195, "y0": 110, "x1": 285, "y1": 320},
        {"score": 0.90, "x0": 185, "y0": 160, "x1": 300, "y1": 360},  # tipping, widening
        {"score": 0.70, "x0": 170, "y0": 230, "x1": 330, "y1": 400},  # low, wide
        None, None,                                                    # prone: detector loses it
    ]


def test_stub_fall_produces_descent_evidence():
    ev = encode_video("clip://fall", backend="stub",
                      stub_track=_upright_then_fall_track(), img_h=512)
    assert isinstance(ev, EvidenceModel)
    assert ev.modality == "vision" and ev.detector == "stub"
    assert ev.reads("person_present", min_conf=0.9)
    # the fall onset is visible as descent even though the prone frame is lost
    assert ev.reads("rapid_descent", min_conf=0.4, min_value=0.5)
    assert ev.reads("on_floor", min_conf=0.4, min_value=0.3)


def test_no_person_abstains_not_vetoes():
    ev = encode_video("clip://empty", backend="stub", stub_track=[None, None, None])
    p = ev.get("person_present")
    assert p is not None and p.confidence == 0.0
    # an abstaining witness corroborates nothing (and does not fabricate a veto)
    assert not ev.reads("person_present")
    assert not ev.reads("rapid_descent")


def test_commitment_hash_roundtrips():
    ev = encode_video("clip://x", backend="stub", stub_track=_upright_then_fall_track())
    assert ev.commitment_hash and ev.verify()
    # tampering with an observable value breaks verification
    ev.observables = list(ev.observables) + [
        PhysicalObservable(name="person_present", value=1.0, confidence=1.0)]
    assert not ev.verify()


def test_stream_catches_fall_in_rolling_window():
    """A live camera: push frames one at a time; the fall shows up in the window
    as the person descends. Uses a scripted detector so no torch is needed."""
    track = _upright_then_fall_track()  # per-frame boxes (or None)

    def detect(frame):
        # frame carries its index in pixel [0,0,0]; return the scripted box
        i = int(frame[0, 0, 0])
        return track[i] if i < len(track) else None

    s = VideoWitnessStream(window=6, detect_fn=detect, img_h=512)
    ev_early = None
    for i in range(len(track)):
        fr = np.zeros((512, 512, 3), dtype=np.uint8)
        fr[0, 0, 0] = i
        s.push(fr)
        if i == 3:  # mid-fall, still tracked
            ev_early = s.evidence()
    assert ev_early.reads("person_present", min_conf=0.6)
    assert ev_early.reads("rapid_descent", min_conf=0.4, min_value=0.4)


def _attested(counter=5, age_s=2, payload="deadbeef"):
    now = datetime.now(timezone.utc)
    att = SensorAttestation(
        device_id="cam-robot-01", key_id="k1", counter=counter,
        signed_at=(now - timedelta(seconds=age_s)).isoformat(),
        payload_sha256=payload, signature="sig",
    )
    return EvidenceModel(
        evidence_id="e1", modality="vision", source="camera://robot",
        source_sha256=payload, attestation=att,
        observables=[PhysicalObservable(name="person_present", value=1.0, confidence=0.95)],
    ).finalize()


def test_attestation_fresh_and_advancing_is_trusted():
    ev = _attested(counter=5, age_s=2)
    ok, why = check_attestation(ev, verify_sig=lambda *_: True,
                                max_age_s=30, min_counter=4)
    assert ok, why


def test_attestation_replay_counter_rejected():
    ev = _attested(counter=4)
    ok, why = check_attestation(ev, verify_sig=lambda *_: True, min_counter=4)
    assert not ok and "counter" in why


def test_attestation_stale_capture_rejected():
    ev = _attested(age_s=120)
    ok, why = check_attestation(ev, verify_sig=lambda *_: True, max_age_s=30)
    assert not ok and "stale" in why


def test_attestation_payload_mismatch_rejected():
    ev = _attested(payload="aaaa")
    ev.source_sha256 = "bbbb"  # signed hash no longer matches the evidence source
    ok, why = check_attestation(ev, verify_sig=lambda *_: True)
    assert not ok and "payload" in why


def test_missing_attestation_abstains_when_required():
    ev = encode_video("clip://x", backend="stub", stub_track=[None])
    ok, why = check_attestation(ev, require=True)
    assert not ok and "no attestation" in why


def test_real_ed25519_roundtrip():
    crypto = pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.exceptions import InvalidSignature

    sk = Ed25519PrivateKey.generate()
    pk = sk.public_key()
    now = datetime.now(timezone.utc)
    payload_hash = hashlib.sha256(b"frames").hexdigest()
    att = SensorAttestation(device_id="cam", key_id="k", counter=10,
                            signed_at=now.isoformat(), payload_sha256=payload_hash)
    att.signature = sk.sign(att.signing_payload()).hex()
    ev = EvidenceModel(evidence_id="e", modality="vision", source="cam",
                       source_sha256=payload_hash, attestation=att).finalize()

    def verify_sig(payload, sig_hex, key_id):
        try:
            pk.verify(bytes.fromhex(sig_hex), payload)
            return True
        except InvalidSignature:
            return False

    ok, why = check_attestation(ev, verify_sig=verify_sig, max_age_s=60, min_counter=9)
    assert ok, why
    # tamper: flip the counter after signing -> signature no longer matches payload
    ev.attestation.counter = 11
    ok2, _ = check_attestation(ev, verify_sig=verify_sig, max_age_s=60)
    assert not ok2
