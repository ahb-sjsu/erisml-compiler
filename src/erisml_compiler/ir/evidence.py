"""EvidenceModel — the erisml IR for physically-witnessed ground truth.

The moral tensor (`MoralTensorV3`) is an *advisory* object: a learned evaluation
of what an action is worth. `EvidenceModel` is the *authority* object: physical
observables witnessed from a real sensor stream (here, video), each with its own
confidence and a commitment hash over the whole record.

The two are deliberately different types. An evaluator can be reasoned into any
verdict; a physical witness reports what a sensor measured. In a governor built
for structural containment, elevated authority is gated on `EvidenceModel`
corroboration (a channel that can GRANT), while the moral tensor can only advise
downward. Keeping them as separate IR types puts that split in the type system.

A witness reports *physical observables* — "a person is present", "the body is
horizontal", "low to the ground", "not moving", "descended rapidly" — never a
judgment like "this is an emergency". Composing observables into a situation
(and gating on it) is the consumer's job, not the witness's. That boundary is
what keeps the gameable reasoning surface out of the authority channel.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field


class PhysicalObservable(BaseModel):
    """One physical fact read from a sensor stream, with the sensor's own confidence.

    `value` is a normalized reading whose meaning is fixed by `name` (e.g.
    body_horizontal in [0,1] where 1 is fully prone; vertical_position in [0,1]
    where 0 is on the floor). `confidence` in [0,1] is how sure the sensor is of
    the reading — it is NOT the value. A witness abstains by reporting low
    confidence, and a corroboration gate simply does not count an abstaining
    witness (it never fabricates one)."""

    model_config = ConfigDict(frozen=True)

    name: str
    value: float
    confidence: float = Field(ge=0.0, le=1.0)
    modality: str = "vision"
    unit: str = ""
    note: str = ""


class SensorAttestation(BaseModel):
    """A hardware attestation of a sensor stream: a signature produced inside the
    device by a key in a secure element (C2PA / Content Credentials, Axis signed
    video, Truepic/Qualcomm, TPM/TEE), over the captured bytes.

    A signature proves origin and integrity (the frames came from this device and
    were not altered after capture); it does NOT prove the scene is real (the
    analog hole). So the gate pairs it with freshness (a monotonic `counter` and a
    trusted `signed_at`, defeating replay) and, above all, corroboration across
    independent, independently-keyed sensors. `payload_sha256` is the hash that was
    actually signed; the verifier checks it matches the evidence it accompanies."""

    device_id: str
    key_id: str = ""  # certificate / public-key identifier (PKI)
    algorithm: str = "ed25519"
    signature: str = ""  # hex/base64 signature over payload_sha256 || counter || signed_at
    counter: int = 0  # monotonic anti-replay counter
    signed_at: str = ""  # trusted capture timestamp (ISO 8601)
    payload_sha256: str = ""  # hash of the bytes/record that were signed
    cert_chain: list[str] = Field(default_factory=list)

    def signing_payload(self) -> bytes:
        """The exact bytes a verifier must check the signature against."""
        return f"{self.payload_sha256}|{self.counter}|{self.signed_at}".encode("utf-8")


class EvidenceModel(BaseModel):
    """A ground-truth model compiled from one sensor stream.

    Carries the physical observables, the detector provenance, a hash of the
    source bytes, an optional hardware attestation, and a commitment hash over the
    canonical record so a witness claim can be re-verified. This is the object a
    governor's witness gate reads."""

    evidence_id: str
    modality: str  # e.g. "vision"
    source: str  # path or camera id
    n_frames: int = 0
    fps: float | None = None
    observables: list[PhysicalObservable] = Field(default_factory=list)
    detector: str = ""  # model name + weights version (provenance)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_sha256: str = ""
    attestation: SensorAttestation | None = None
    commitment_hash: str = ""

    def get(self, name: str) -> PhysicalObservable | None:
        for o in self.observables:
            if o.name == name:
                return o
        return None

    def reads(self, name: str, min_conf: float = 0.5, min_value: float = 0.5) -> bool:
        """True iff the named observable is present at adequate confidence AND its
        value clears `min_value`. A witness that abstains (low confidence) reads
        False — it does not corroborate, and it does not veto either."""
        o = self.get(name)
        return o is not None and o.confidence >= min_conf and o.value >= min_value

    def canonical(self) -> dict:
        d = self.model_dump(mode="json")
        d.pop("commitment_hash", None)
        return d

    def finalize(self) -> "EvidenceModel":
        """Compute the commitment hash over the canonical record (excluding the
        hash field itself). Two identical witnessings hash identically."""
        blob = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"))
        self.commitment_hash = hashlib.sha256(blob.encode("utf-8")).hexdigest()
        return self

    def verify(self) -> bool:
        blob = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"))
        return self.commitment_hash == hashlib.sha256(blob.encode("utf-8")).hexdigest()


def check_attestation(
    ev: EvidenceModel,
    *,
    verify_sig=None,
    now: datetime | None = None,
    max_age_s: float | None = None,
    min_counter: int | None = None,
    require: bool = True,
) -> tuple[bool, str]:
    """Decide whether an EvidenceModel's stream may be TRUSTED as a witness.

    Checks, in order: an attestation is present (if required); the signature
    verifies (via `verify_sig(payload_bytes, signature, key_id) -> bool`, so the
    crypto backend is pluggable — supply an ed25519 verifier in production);
    the signed payload hash matches this evidence's source; the capture is fresh
    (`max_age_s`, anti-replay); and the monotonic counter has advanced
    (`min_counter`). Returns (ok, reason). A stream that fails any check is NOT
    trusted — the witness abstains; it never becomes a veto.
    """
    a = ev.attestation
    if a is None:
        return (not require), ("no attestation" if require else "attestation not required")
    if verify_sig is not None:
        try:
            if not verify_sig(a.signing_payload(), a.signature, a.key_id):
                return False, "signature did not verify"
        except Exception as e:  # a broken verifier fails closed
            return False, f"signature verify error: {type(e).__name__}"
    if a.payload_sha256 and ev.source_sha256 and a.payload_sha256 != ev.source_sha256:
        return False, "signed payload hash does not match evidence source"
    if min_counter is not None and a.counter <= min_counter:
        return False, f"stale/replayed counter ({a.counter} <= {min_counter})"
    if max_age_s is not None:
        if not a.signed_at:
            return False, "no signed_at timestamp for freshness check"
        try:
            ts = datetime.fromisoformat(a.signed_at)
            ref = now or datetime.now(timezone.utc)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age = (ref - ts).total_seconds()
            if age > max_age_s:
                return False, f"stale capture ({age:.0f}s > {max_age_s}s)"
        except ValueError:
            return False, "unparseable signed_at"
    return True, "attested, fresh, counter advanced"
