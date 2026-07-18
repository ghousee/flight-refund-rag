"""Convert the ``data/raw/`` corpus into markdown notes in the Obsidian vault.

Two paths, each playing to the source format's strengths:

  * Airline tariffs & the APPR (**PDF**, via PyMuPDF) -> ONE note per document.
    PDF section detection is unreliable, so we keep the whole document as a
    single note and let ``index.py`` do the retrieval-sized chunking later.
  * US DOT regulations (**eCFR XML**) -> ONE note per section (e.g. § 260.6).
    The XML has clean ``<DIV8>`` section boundaries, giving neatly scoped notes
    with accurate per-section ``topic`` metadata.

Every note gets YAML frontmatter auto-stamped from ``data/raw/manifest.json``
so ``source_url`` and ``retrieved_date`` provenance is mechanical, not manual.

Output layout (see .gitignore — airline notes are NOT committed):
    vault/airlines/      Air Canada, WestJet    (gitignored)
    vault/regulations/   APPR, US DOT           (committed)

Usage:
    python -m src.ingest.parse_pdfs
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import fitz  # PyMuPDF
import frontmatter

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
VAULT_DIR = ROOT / "vault"
MANIFEST_PATH = RAW_DIR / "manifest.json"

# category -> vault subfolder
CATEGORY_DIR = {"airline": "airlines", "regulation": "regulations"}


def _slugify(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^\w\s-]", "", text.lower())
    text = re.sub(r"[\s_-]+", "-", text).strip("-")
    return text[:max_len].strip("-")


def _clean_ws(text: str) -> str:
    """Collapse the ragged whitespace/newlines that XML text nodes carry."""
    return re.sub(r"\s+", " ", text).strip()


def _base_meta(entry: dict) -> dict:
    """Frontmatter fields common to every note, drawn from the manifest."""
    meta = {
        "title": entry["title"],
        "jurisdiction": entry["jurisdiction"],
        "doc_type": entry["doc_type"],
        "source_url": entry["url"],
        "retrieved_date": entry["retrieved_date"],
    }
    if entry.get("airline"):
        meta["airline"] = entry["airline"]
    return meta


def _provenance_footer(entry: dict) -> str:
    """Reproduction/provenance note appended to government regulation notes.

    Satisfies the Reproduction of Federal Law Order's "not an official version"
    requirement for Canadian regs, and records the public-domain status of US
    eCFR text. Airline notes get no footer (they are gitignored, not published).
    """
    if entry["category"] != "regulation":
        return ""
    url, date = entry["url"], entry["retrieved_date"]
    if entry["jurisdiction"] == "Canada":
        return (
            f"\n\n---\n> Reproduced from the Justice Laws website ({url}) as "
            f"retrieved {date}. This is **not an official version**."
        )
    return (
        f"\n\n---\n> Reproduced from the U.S. eCFR ({url}) as retrieved {date}. "
        f"U.S. Government work — public domain."
    )


def _write_note(out_dir: Path, slug: str, meta: dict, body: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    post = frontmatter.Post(body.strip() + "\n", **meta)
    path = out_dir / f"{slug}.md"
    path.write_bytes(frontmatter.dumps(post).encode("utf-8"))
    return path


# ---------------------------------------------------------------------------
# PDF -> one note per document
# ---------------------------------------------------------------------------
def parse_pdf(entry: dict, out_dir: Path) -> list[Path]:
    src = RAW_DIR / entry["filename"]
    with fitz.open(src) as doc:
        pages = [page.get_text("text") for page in doc]
    # Join pages, then tidy: trim trailing spaces and collapse blank-line runs.
    text = "\n\n".join(pages)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    meta = _base_meta(entry)
    meta.setdefault("topic", entry["doc_type"])
    body = f"# {entry['title']}\n\n{text}" + _provenance_footer(entry)
    slug = _slugify(Path(entry["filename"]).stem)
    return [_write_note(out_dir, slug, meta, body)]


# ---------------------------------------------------------------------------
# eCFR XML -> one note per <DIV8> section
# ---------------------------------------------------------------------------
def parse_ecfr_xml(entry: dict, out_dir: Path) -> list[Path]:
    tree = ET.parse(RAW_DIR / entry["filename"])
    root = tree.getroot()
    written: list[Path] = []

    for section in root.iter("DIV8"):
        number = section.get("N", "").strip()  # e.g. "260.6"
        head_el = section.find("HEAD")
        head = _clean_ws("".join(head_el.itertext())) if head_el is not None else number

        # Body: every paragraph in document order (P/FP), plus the citation note.
        paras = [
            _clean_ws("".join(p.itertext()))
            for p in section.iter()
            if p.tag in {"P", "FP"}
        ]
        paras = [p for p in paras if p]
        cita = section.find("CITA")
        if cita is not None:
            paras.append(f"_{_clean_ws(''.join(cita.itertext()))}_")
        if not paras:
            continue

        # topic = the section title without its "§ 260.6" citation prefix.
        topic = re.sub(r"^§?\s*[\d.]+\s*", "", head).strip().rstrip(".")

        meta = _base_meta(entry)
        meta["citation"] = f"14 CFR {number}"
        meta["topic"] = topic
        body = f"# {head}\n\n" + "\n\n".join(paras) + _provenance_footer(entry)
        slug = _slugify(f"14-cfr-{number.replace('.', '-')}-{topic}")
        written.append(_write_note(out_dir, slug, meta, body))

    return written


def parse_all() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    total = 0
    for filename, entry in manifest.items():
        entry = {**entry, "filename": filename}
        out_dir = VAULT_DIR / CATEGORY_DIR[entry["category"]]
        if entry["fmt"] == "pdf":
            notes = parse_pdf(entry, out_dir)
        elif entry["fmt"] == "xml":
            notes = parse_ecfr_xml(entry, out_dir)
        else:
            raise ValueError(f"{filename}: unsupported fmt {entry['fmt']!r}")
        total += len(notes)
        rel = out_dir.relative_to(ROOT)
        print(f"  {filename:38s} -> {len(notes):2d} note(s) in {rel}/")
    print(f"\nWrote {total} note(s) to the vault.")


if __name__ == "__main__":
    parse_all()
