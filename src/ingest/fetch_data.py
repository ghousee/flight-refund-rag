"""Download the flight-refund corpus from official sources into ``data/raw/``.

Legal / content rules this script enforces:
  * It downloads ONLY the specific official URLs listed in ``SOURCES`` below.
    No crawling, no bulk scraping — one-time downloads of named documents.
  * Airline-authored files (Air Canada, WestJet tariffs) land in ``data/raw/``,
    which is gitignored. They are never committed.
  * Every download is recorded in ``data/raw/manifest.json`` with its
    ``source_url``, ``retrieved_date`` and a SHA-256 hash, so ``parse_pdfs.py``
    can stamp traceable provenance into each note's frontmatter.

Usage:
    python -m src.ingest.fetch_data          # download anything missing
    python -m src.ingest.fetch_data --force  # re-download everything
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
MANIFEST_PATH = RAW_DIR / "manifest.json"

# A real browser User-Agent — some official CDNs (airline sites) reject the
# default python-requests agent.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# eCFR "point in time" version. The versioner API returns the regulation text
# in effect on this date. Pinned for reproducibility; bump it to re-pull.
ECFR_DATE = "2026-07-01"


@dataclass(frozen=True)
class Source:
    """One official document to download.

    The static metadata here (airline, jurisdiction, doc_type, ...) flows into
    each note's YAML frontmatter during parsing.
    """

    id: str
    url: str
    filename: str
    fmt: str            # "pdf" | "xml"
    category: str       # "airline" (gitignored notes) | "regulation" (committed)
    doc_type: str       # "tariff" | "regulation"
    jurisdiction: str   # "Canada" | "US"
    title: str
    airline: str | None = None


# The complete, explicit corpus. Adding a source here is the ONLY way the
# script downloads anything — there is no discovery/crawling.
SOURCES: list[Source] = [
    Source(
        id="air-canada-domestic-tariff",
        url="https://www.aircanada.com/content/dam/aircanada/portal/documents/PDF/en/ac_domestic_tariff_en.pdf",
        filename="air_canada_domestic_tariff.pdf",
        fmt="pdf",
        category="airline",
        doc_type="tariff",
        jurisdiction="Canada",
        title="Air Canada Domestic Tariff (General Rules)",
        airline="Air Canada",
    ),
    Source(
        id="air-canada-international-tariff",
        url="https://www.aircanada.com/content/dam/aircanada/portal/documents/PDF/en/International_Tariff_en.pdf",
        filename="air_canada_international_tariff.pdf",
        fmt="pdf",
        category="airline",
        doc_type="tariff",
        jurisdiction="Canada",
        title="Air Canada International Tariff (General Rules)",
        airline="Air Canada",
    ),
    Source(
        # NOTE: WestJet versions this filename by effective date, so the URL
        # changes when they publish a new tariff. Update it here when it 404s.
        # (Their CDN mislabels the content-type as text/html, but the bytes are
        # a real PDF — we validate by magic number below, not by header.)
        id="westjet-international-tariff",
        url="https://www.westjet.com/content/dam/westjet/documents/en/tariffs/WSI_EN_FE_2026-04-23.pdf",
        filename="westjet_international_tariff.pdf",
        fmt="pdf",
        category="airline",
        doc_type="tariff",
        jurisdiction="Canada",
        title="WestJet International/Transborder Tariff (WS1, CTA No. 518)",
        airline="WestJet",
    ),
    Source(
        id="appr-sor-2019-150",
        url="https://laws-lois.justice.gc.ca/PDF/SOR-2019-150.pdf",
        filename="appr_sor_2019-150.pdf",
        fmt="pdf",
        category="regulation",
        doc_type="regulation",
        jurisdiction="Canada",
        title="Air Passenger Protection Regulations (SOR/2019-150)",
    ),
    Source(
        # Part 260 is the CORE refund rule: the 2024 DOT final rule created this
        # part specifically for automatic refunds (what/when/how refunds are owed).
        id="us-dot-14-cfr-260",
        url=f"https://www.ecfr.gov/api/versioner/v1/full/{ECFR_DATE}/title-14.xml?part=260",
        filename="us_dot_14_cfr_260.xml",
        fmt="xml",
        category="regulation",
        doc_type="regulation",
        jurisdiction="US",
        title="US DOT — 14 CFR Part 260 (Refunds and Other Consumer Protections)",
    ),
    Source(
        # Part 259 (Enhanced Protections) — customer service plans, tarmac delays,
        # and refund-adjacent commitments that cross-reference Part 260.
        id="us-dot-14-cfr-259",
        url=f"https://www.ecfr.gov/api/versioner/v1/full/{ECFR_DATE}/title-14.xml?part=259",
        filename="us_dot_14_cfr_259.xml",
        fmt="xml",
        category="regulation",
        doc_type="regulation",
        jurisdiction="US",
        title="US DOT — 14 CFR Part 259 (Enhanced Protections for Airline Passengers)",
    ),
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def _validate(path: Path, fmt: str) -> None:
    """Cheap sanity check that we downloaded the right kind of file."""
    head = path.read_bytes()[:512].lstrip()
    if fmt == "pdf" and not head.startswith(b"%PDF"):
        raise ValueError(f"{path.name}: expected a PDF but got {head[:32]!r}")
    if fmt == "xml" and not head.startswith(b"<"):
        raise ValueError(f"{path.name}: expected XML but got {head[:32]!r}")


def _download(src: Source, session: requests.Session) -> Path:
    dest = RAW_DIR / src.filename
    resp = session.get(src.url, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    _validate(dest, src.fmt)
    return dest


def fetch(force: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    manifest: dict[str, dict] = {}
    today = date.today().isoformat()

    for src in SOURCES:
        dest = RAW_DIR / src.filename
        if dest.exists() and not force:
            print(f"  skip   {src.filename} (already present)")
        else:
            print(f"  get    {src.filename}  <-  {src.url}")
            dest = _download(src, session)
            print(f"         {dest.stat().st_size / 1_048_576:.1f} MB")

        manifest[src.filename] = {
            **{k: v for k, v in asdict(src).items() if k != "filename"},
            "retrieved_date": today,
            "sha256": _sha256(dest),
            "size_bytes": dest.stat().st_size,
        }

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote manifest for {len(manifest)} source(s) -> {MANIFEST_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="re-download files that already exist"
    )
    args = parser.parse_args()
    fetch(force=args.force)


if __name__ == "__main__":
    main()
