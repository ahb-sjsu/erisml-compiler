"""Wire a freshly-minted Zenodo DOI into a repo's CITATION.cff + README badge.

Idempotent: re-running with the same DOI is a no-op. Updating an existing
CITATION.cff preserves any custom fields the user has already added.

Usage:
    python wire_zenodo_doi.py <repo-path> <concept-doi> <version-doi> \
        --title "Title" --version "X.Y.Z" --date "YYYY-MM-DD"

Where DOIs are bare (e.g. 10.5281/zenodo.12345678), not URLs.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


CITATION_TEMPLATE = """cff-version: 1.2.0
message: "If you use this software in academic work, please cite via the Zenodo DOI below."
title: "{title}"
authors:
  - family-names: Bond
    given-names: Andrew H.
    email: andrew.bond@sjsu.edu
    affiliation: "San Jose State University"
    orcid: "https://orcid.org/0009-0009-1769-5099"
type: software
version: "{version}"
date-released: "{date}"
license: MIT
repository-code: "{repo_url}"
url: "{repo_url}"
doi: "{concept_doi}"
identifiers:
  - description: "This specific release (v{version})"
    type: doi
    value: "{version_doi}"
  - description: "Concept DOI (always resolves to the latest release)"
    type: doi
    value: "{concept_doi}"
"""

BADGE_PATTERN = re.compile(r"\[!\[DOI\][^\]]*\]\([^)]*zenodo[^)]*\)")


def make_badge(concept_doi: str) -> str:
    return (
        f"[![DOI](https://zenodo.org/badge/DOI/{concept_doi}.svg)]"
        f"(https://doi.org/{concept_doi})"
    )


def update_or_create_citation(
    repo_path: Path, title: str, version: str, date: str,
    concept_doi: str, version_doi: str, repo_url: str,
) -> str:
    cff = repo_path / "CITATION.cff"
    if not cff.exists():
        content = CITATION_TEMPLATE.format(
            title=title, version=version, date=date,
            concept_doi=concept_doi, version_doi=version_doi, repo_url=repo_url,
        )
        cff.write_text(content, encoding="utf-8")
        return "created"

    text = cff.read_text(encoding="utf-8")
    # If a doi field already exists, replace it; otherwise append.
    if re.search(r"^doi:\s*", text, re.MULTILINE):
        text = re.sub(r"^doi:.*$", f'doi: "{concept_doi}"', text, count=1, flags=re.MULTILINE)
    else:
        # Find a sensible insertion point — after license:, fallback to end.
        if re.search(r"^license:", text, re.MULTILINE):
            text = re.sub(
                r"(^license:.*$)",
                rf'\1\ndoi: "{concept_doi}"',
                text, count=1, flags=re.MULTILINE,
            )
        else:
            text = text.rstrip() + f'\ndoi: "{concept_doi}"\n'

    # Add identifiers block if missing.
    if "identifiers:" not in text:
        identifiers_block = f"""identifiers:
  - description: "This specific release (v{version})"
    type: doi
    value: "{version_doi}"
  - description: "Concept DOI (always resolves to the latest release)"
    type: doi
    value: "{concept_doi}"
"""
        text = text.rstrip() + "\n" + identifiers_block

    cff.write_text(text, encoding="utf-8")
    return "updated"


def update_readme_badge(repo_path: Path, concept_doi: str) -> str:
    readme = repo_path / "README.md"
    if not readme.exists():
        return "no-readme"
    text = readme.read_text(encoding="utf-8")
    new_badge = make_badge(concept_doi)
    if new_badge in text:
        return "already-present"
    # Replace any existing zenodo badge.
    if BADGE_PATTERN.search(text):
        text = BADGE_PATTERN.sub(new_badge, text, count=1)
        readme.write_text(text, encoding="utf-8")
        return "replaced"
    # Else insert after the first line (title) so it sits in the header.
    lines = text.split("\n", 1)
    if len(lines) == 2:
        # Place under existing badge block if present, otherwise right after H1.
        header, rest = lines
        # Walk rest until we find a non-badge non-blank line.
        rest_lines = rest.split("\n")
        i = 0
        # skip leading blank lines
        while i < len(rest_lines) and rest_lines[i].strip() == "":
            i += 1
        # Walk through existing badges
        while i < len(rest_lines) and rest_lines[i].strip().startswith("[![") and "]" in rest_lines[i]:
            i += 1
        # Insert the badge before `rest_lines[i]`
        rest_lines.insert(i, new_badge)
        new_text = header + "\n\n" + "\n".join(rest_lines).lstrip("\n")
        readme.write_text(new_text, encoding="utf-8")
        return "inserted"
    return "no-anchor"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_path", type=Path)
    ap.add_argument("concept_doi")
    ap.add_argument("version_doi")
    ap.add_argument("--title", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--repo-url", required=True)
    args = ap.parse_args()

    cff_status = update_or_create_citation(
        args.repo_path, args.title, args.version, args.date,
        args.concept_doi, args.version_doi, args.repo_url,
    )
    badge_status = update_readme_badge(args.repo_path, args.concept_doi)
    print(f"CITATION.cff: {cff_status}")
    print(f"README badge: {badge_status}")


if __name__ == "__main__":
    main()
