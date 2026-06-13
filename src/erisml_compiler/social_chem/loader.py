"""Read the Social Chem 101 TSV and group rows by situation.

Source: https://maxwellforbes.com/social-chemistry/
License: CC-BY 4.0 (per the Social Chem 101 README and EMNLP 2020 paper).

The TSV is 25 columns wide; we keep only the ones we use at fit time
(see `schema.SituationRoT`).
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

from erisml_compiler.social_chem.schema import (
    CorpusFingerprint,
    Situation,
    SituationRoT,
)

EXPECTED_COLUMNS: tuple[str, ...] = (
    "area",
    "m",
    "split",
    "rot-agree",
    "rot-categorization",
    "rot-moral-foundations",
    "rot-char-targeting",
    "rot-bad",
    "rot-judgment",
    "action",
    "action-agency",
    "action-moral-judgment",
    "action-agree",
    "action-legal",
    "action-pressure",
    "action-char-involved",
    "action-hypothetical",
    "situation",
    "situation-short-id",
    "rot",
    "rot-id",
    "rot-worker-id",
    "breakdown-worker-id",
    "n-characters",
    "characters",
)

CITATION = (
    "Forbes, Hwang, Shwartz, Sap, Choi. Social Chemistry 101: Learning to "
    "Reason about Social and Moral Norms. EMNLP 2020. "
    "https://aclanthology.org/2020.emnlp-main.48/"
)


def _parse_int(v: str) -> int | None:
    v = (v or "").strip()
    if not v:
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def iter_rows(
    tsv_path: str | Path,
    *,
    area: str | None = None,
    limit: int | None = None,
) -> Iterator[SituationRoT]:
    """Yield SituationRoT objects from a Social Chem 101 TSV.

    `area` filters rows by the `area` column (e.g. 'dearabby'). When
    None, all areas pass through (useful when reading a pre-extracted
    single-area slice).
    """
    path = Path(tsv_path)
    if not path.exists():
        raise FileNotFoundError(f"Social Chem 101 TSV not found at {path}")

    yielded = 0
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        cols = tuple(reader.fieldnames or ())
        if cols != EXPECTED_COLUMNS:
            raise ValueError(
                f"Unexpected TSV columns at {path}: got {cols}, " f"expected {EXPECTED_COLUMNS}"
            )

        for row in reader:
            row_area = (row.get("area") or "").strip()
            if area is not None and row_area != area:
                continue

            mft_raw = (row.get("rot-moral-foundations") or "").strip()
            mft = tuple(p.strip() for p in mft_raw.split("|") if p.strip()) if mft_raw else ()

            yield SituationRoT(
                area=row_area,
                situation_short_id=(row.get("situation-short-id") or "").strip(),
                situation=(row.get("situation") or "").strip(),
                rot=(row.get("rot") or "").strip(),
                rot_categorization=(row.get("rot-categorization") or "").strip(),
                rot_moral_foundations=mft,
                rot_agree=_parse_int(row.get("rot-agree") or ""),
                action=(row.get("action") or "").strip(),
                action_moral_judgment=_parse_int(row.get("action-moral-judgment") or ""),
                action_agree=_parse_int(row.get("action-agree") or ""),
            )
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def group_by_situation(rots: Iterable[SituationRoT]) -> list[Situation]:
    """Group RoT rows into Situations keyed by situation_short_id."""
    buckets: dict[str, list[SituationRoT]] = defaultdict(list)
    titles: dict[str, str] = {}
    areas: dict[str, str] = {}
    for r in rots:
        if not r.situation_short_id:
            continue
        buckets[r.situation_short_id].append(r)
        if r.situation_short_id not in titles and r.situation:
            titles[r.situation_short_id] = r.situation
        if r.situation_short_id not in areas:
            areas[r.situation_short_id] = r.area

    out: list[Situation] = []
    for sid in sorted(buckets):
        out.append(
            Situation(
                situation_short_id=sid,
                situation=titles.get(sid, ""),
                area=areas.get(sid, ""),
                rots=tuple(buckets[sid]),
            )
        )
    return out


def load_situations(
    tsv_path: str | Path,
    *,
    area: str | None = None,
    limit: int | None = None,
) -> list[Situation]:
    """Convenience: iter_rows -> group_by_situation."""
    return group_by_situation(iter_rows(tsv_path, area=area, limit=limit))


def fingerprint_corpus(
    situations: list[Situation],
    *,
    source: str,
    license: str = "CC-BY 4.0",
    citation: str = CITATION,
) -> CorpusFingerprint:
    """Deterministic fingerprint over a corpus slice.

    Hash is over sorted (situation_short_id, n_rots, ordered rot text)
    tuples. Order-independent across the input.
    """
    sorted_sits = sorted(situations, key=lambda s: s.situation_short_id)
    canonical = json.dumps(
        [
            {
                "sid": s.situation_short_id,
                "n": len(s.rots),
                "rots": sorted(r.rot.strip().lower() for r in s.rots),
            }
            for s in sorted_sits
        ],
        sort_keys=True,
        separators=(",", ":"),
    )
    h = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    found_dist: dict[str, int] = defaultdict(int)
    judg_dist: dict[str, int] = defaultdict(int)
    n_rows = 0
    for s in situations:
        for r in s.rots:
            n_rows += 1
            for f in r.rot_moral_foundations:
                found_dist[f] += 1
            if r.action_moral_judgment is not None:
                judg_dist[str(r.action_moral_judgment)] += 1

    return CorpusFingerprint(
        source=source,
        license=license,
        citation=citation,
        schema_tsv_columns=list(EXPECTED_COLUMNS),
        n_rows=n_rows,
        n_situations=len(situations),
        canonical_sha256=h,
        foundation_distribution=dict(found_dist),
        judgment_distribution=dict(judg_dist),
    )
