"""JOSS pre-submission checklist verifier.

Runs over `paper/paper.md` + `paper/paper.bib` + the repo root and
fails noisily if any required item is missing or malformed. Designed
to be invoked as a workflow step before the JOSS PDF build, so a
broken submission surfaces at PR-time rather than at JOSS-bot time.

The checklist mirrors the JOSS submission requirements
(https://joss.readthedocs.io/en/latest/submitting.html):

  Repo-level:
    - LICENSE file present at repo root (or in standard location)
    - paper/paper.md present
    - paper/paper.bib present

  paper.md YAML frontmatter:
    - title (non-empty)
    - tags (list, non-empty)
    - authors (list with at least one entry; each entry has 'name';
      at least one entry has 'orcid')
    - affiliations (list, indexed)
    - date (non-empty string)
    - bibliography (resolves to an existing file relative to paper/)

  paper.md body:
    - "# Summary" section
    - "# Statement of need" section
    - "# References" section (just the heading; pandoc fills the body)

  paper.bib:
    - File is parseable (basic BibTeX validation)
    - At least one entry (so 'References' is non-vacuous)

Exits 0 on success, 1 (with a structured summary) on any failure.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = REPO_ROOT / "paper"
PAPER_MD = PAPER_DIR / "paper.md"
PAPER_BIB = PAPER_DIR / "paper.bib"


class CheckFailure(Exception):
    """Raised by individual checks to signal a problem."""


def _check_repo_files() -> list[str]:
    """Verify repo-level files JOSS expects exist."""
    errors: list[str] = []

    # LICENSE in repo root, or one of the standard names.
    license_candidates = ["LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING"]
    if not any((REPO_ROOT / n).exists() for n in license_candidates):
        errors.append(
            f"No license file at repo root (tried {license_candidates})"
        )

    if not PAPER_MD.exists():
        errors.append(f"Missing {PAPER_MD.relative_to(REPO_ROOT)}")
    if not PAPER_BIB.exists():
        errors.append(f"Missing {PAPER_BIB.relative_to(REPO_ROOT)}")

    return errors


def _split_frontmatter(text: str) -> tuple[dict, str]:
    """Parse the YAML frontmatter at the top of paper.md.

    Returns (frontmatter_dict, body). The frontmatter is delimited
    by lines `---` and `---` at the start of the file.
    """
    import yaml

    if not text.startswith("---"):
        raise CheckFailure(
            "paper.md does not start with YAML frontmatter (`---`)"
        )
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise CheckFailure("paper.md has malformed YAML frontmatter")
    fm = yaml.safe_load(parts[1]) or {}
    body = parts[2]
    return fm, body


def _check_frontmatter(fm: dict) -> list[str]:
    errors: list[str] = []

    title = fm.get("title") or ""
    if not str(title).strip():
        errors.append("frontmatter: `title` is missing or empty")
    tags = fm.get("tags")
    if not isinstance(tags, list) or not tags:
        errors.append("frontmatter: `tags` must be a non-empty list")
    authors = fm.get("authors")
    if not isinstance(authors, list) or not authors:
        errors.append("frontmatter: `authors` must be a non-empty list")
    else:
        any_orcid = False
        for i, a in enumerate(authors):
            if not isinstance(a, dict):
                errors.append(f"frontmatter: authors[{i}] must be a dict")
                continue
            if not str(a.get("name") or "").strip():
                errors.append(f"frontmatter: authors[{i}].name missing")
            if a.get("orcid"):
                any_orcid = True
                orcid = a["orcid"]
                # Basic ORCID shape check: 4×4 hex/digit groups with dashes.
                if not re.match(r"^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$", str(orcid)):
                    errors.append(
                        f"frontmatter: authors[{i}].orcid '{orcid}' is "
                        f"not in 0000-0000-0000-0000 form"
                    )
            if "affiliation" not in a:
                errors.append(
                    f"frontmatter: authors[{i}] has no `affiliation` index"
                )
        if not any_orcid:
            errors.append(
                "frontmatter: at least one author must have an `orcid` field"
            )

    affiliations = fm.get("affiliations")
    if not isinstance(affiliations, list) or not affiliations:
        errors.append("frontmatter: `affiliations` must be a non-empty list")
    else:
        for i, af in enumerate(affiliations):
            if not isinstance(af, dict):
                errors.append(f"frontmatter: affiliations[{i}] must be a dict")
                continue
            if "name" not in af:
                errors.append(f"frontmatter: affiliations[{i}].name missing")
            if "index" not in af:
                errors.append(f"frontmatter: affiliations[{i}].index missing")

    if not fm.get("date"):
        errors.append("frontmatter: `date` is missing")

    bib = fm.get("bibliography")
    if not bib:
        errors.append("frontmatter: `bibliography` is missing")
    else:
        # Resolve relative to paper/
        bib_path = (PAPER_DIR / bib).resolve()
        if not bib_path.exists():
            errors.append(
                f"frontmatter: bibliography '{bib}' resolves to "
                f"{bib_path}, which does not exist"
            )

    return errors


def _check_body_sections(body: str) -> list[str]:
    """JOSS requires at least Summary + Statement of need + References."""
    errors: list[str] = []
    required = [
        r"^#\s+Summary\b",
        r"^#\s+Statement of need\b",
        r"^#\s+References\b",
    ]
    for pattern in required:
        if not re.search(pattern, body, flags=re.MULTILINE):
            errors.append(
                f"body: missing required section heading "
                f"`{pattern.removeprefix('^').removesuffix(r'\b')}`"
            )
    return errors


def _check_bibtex(bib_path: Path) -> list[str]:
    """Lightweight BibTeX sanity check: file parses + has ≥1 entry."""
    errors: list[str] = []
    if not bib_path.exists():
        return [f"BibTeX file {bib_path} does not exist"]
    text = bib_path.read_text(encoding="utf-8")
    # Count @entry-type{ blocks (excluding @comment / @preamble / @string).
    entries = re.findall(
        r"@(?!comment\b|preamble\b|string\b)(\w+)\s*\{", text, flags=re.IGNORECASE
    )
    if not entries:
        errors.append(f"BibTeX file {bib_path.name} has no entries")

    # Bracket balance (cheap, not bulletproof).
    open_braces = text.count("{")
    close_braces = text.count("}")
    if open_braces != close_braces:
        errors.append(
            f"BibTeX file {bib_path.name} has unbalanced braces "
            f"({open_braces} open vs {close_braces} close)"
        )
    return errors


def main() -> int:
    print("=== JOSS pre-submission checks ===")
    all_errors: list[str] = []

    repo_errors = _check_repo_files()
    all_errors.extend(repo_errors)
    if repo_errors:
        # Don't try frontmatter checks if paper.md is missing.
        if any("paper.md" in e for e in repo_errors):
            _print_results(all_errors)
            return 1

    try:
        fm, body = _split_frontmatter(PAPER_MD.read_text(encoding="utf-8"))
    except CheckFailure as e:
        all_errors.append(str(e))
        _print_results(all_errors)
        return 1

    all_errors.extend(_check_frontmatter(fm))
    all_errors.extend(_check_body_sections(body))
    all_errors.extend(_check_bibtex(PAPER_BIB))

    _print_results(all_errors)
    return 1 if all_errors else 0


def _print_results(errors: list[str]) -> None:
    if not errors:
        print("[OK] all JOSS pre-submission checks passed.")
        return
    print(f"[FAIL] {len(errors)} JOSS check(s) failed:", file=sys.stderr)
    for i, e in enumerate(errors, 1):
        print(f"  {i}. {e}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
