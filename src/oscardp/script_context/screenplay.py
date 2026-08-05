from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable


SCENE_RE = re.compile(
    r"^\s*(?:(?P<lead>\d+[A-Z]?)\s+)?(?P<slug>(?:INT\./EXT\.|INT/EXT\.?|INT\.|EXT\.|I/E\.|EST\.)\s*[^\n]*?)(?:\s+(?P<trail>\d+[A-Z]?))?\s*$",
    re.IGNORECASE,
)
TIME_WORDS = {
    "DAY", "NIGHT", "MORNING", "EVENING", "AFTERNOON", "DAWN", "DUSK",
    "LATER", "CONTINUOUS", "MOMENTS LATER", "SAME TIME",
}
CUE_SUFFIX_RE = re.compile(r"\s*\((?:CONT['’]?D|CONTINUED|V\.?O\.?|O\.?S\.?|OFF)\)\s*$", re.I)
PAGE_NOISE_RE = re.compile(r"^(?:CONTINUED:?|\"?BLUE MOON\"?\s+CONFORMED SCRIPT.*|\d+[A-Z]?)$", re.I)


def stable_scene_id(raw: str) -> str:
    raw = raw.strip().upper()
    match = re.fullmatch(r"(\d+)([A-Z]?)", raw)
    if match:
        return f"scene_{int(match.group(1)):03d}{match.group(2)}"
    safe = re.sub(r"[^A-Z0-9]+", "_", raw).strip("_")
    return f"scene_{safe}"


def normalize_character_cue(cue: str) -> str:
    return CUE_SUFFIX_RE.sub("", cue.strip()).strip()


def is_broken_page(lines: Iterable[str]) -> bool:
    material = [line.strip() for line in lines if line.strip()]
    if not material:
        return True
    single = sum(len(re.sub(r"\W", "", line)) <= 1 for line in material)
    normal = sum(len(line.split()) >= 2 for line in material)
    return len(material) >= 15 and (single / len(material) > 0.45 or normal == 0)


def _split_slugline(slugline: str) -> tuple[str, str, str | None]:
    clean = re.sub(r"\s+", " ", slugline.strip()).rstrip(".")
    prefix = re.match(r"^(INT\.?/EXT\.?|INT\.?|EXT\.?|I/E\.?|EST\.?)\s*", clean, re.I)
    int_ext = prefix.group(1).upper().replace(".", "") if prefix else ""
    rest = clean[prefix.end():].strip() if prefix else clean
    location, sep, ending = rest.rpartition(" - ")
    if sep and ending.upper() in TIME_WORDS:
        return int_ext, location.strip(), ending.upper()
    return int_ext, rest, None


def _looks_like_cue(text: str, x0: float, width: float) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped) > 45 or stripped.startswith("("):
        return False
    if SCENE_RE.match(stripped) or stripped.endswith(('.', '!', '?', ':')):
        return False
    letters = re.sub(r"[^A-Za-z]", "", stripped)
    return bool(letters) and stripped == stripped.upper() and x0 >= width * 0.28


def parse_layout_pages(pages: list[dict[str, Any]], movie_key: str, title: str, source_files: dict[str, str]) -> dict[str, Any]:
    scenes: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    pending_parenthetical: str | None = None
    current_cue: tuple[str, str] | None = None
    broken_pages: list[int] = []
    unnumbered = 0

    for page in pages:
        page_no = int(page["page"])
        entries = sorted(page["lines"], key=lambda row: (round(float(row.get("y0", 0)), 1), float(row.get("x0", 0))))
        texts = [str(row["text"]).strip() for row in entries if str(row.get("text", "")).strip()]
        broken = is_broken_page(texts)
        numeric_by_y = [
            (float(row.get("y0", 0)), str(row["text"]).strip().upper())
            for row in entries if re.fullmatch(r"\d+[A-Z]?", str(row.get("text", "")).strip(), re.I)
        ]
        if broken:
            broken_pages.append(page_no)
        for entry in entries:
            text = re.sub(r"\s+", " ", str(entry.get("text", "")).strip())
            if not text or PAGE_NOISE_RE.fullmatch(text):
                continue
            heading = SCENE_RE.match(text)
            if heading:
                raw_no = (heading.group("lead") or heading.group("trail") or "").upper()
                if not raw_no:
                    same_line = [value for y, value in numeric_by_y if abs(y - float(entry.get("y0", 0))) <= 2.5]
                    if same_line:
                        raw_no = same_line[0]
                slugline = heading.group("slug").strip()
                if not raw_no:
                    unnumbered += 1
                    raw_no = f"UNNUMBERED_{unnumbered:03d}"
                scene_id = stable_scene_id(raw_no)
                if any(scene["scene_id"] == scene_id for scene in scenes):
                    scene_id = f"{scene_id}_{sum(s['scene_id'].startswith(scene_id) for s in scenes) + 1}"
                int_ext, location, time_of_day = _split_slugline(slugline)
                current = {
                    "scene_id": scene_id, "screenplay_scene_id": raw_no,
                    "slugline": slugline, "int_ext": int_ext, "location": location,
                    "time_of_day": time_of_day, "script_pages": {"start": page_no, "end": page_no},
                    "scene_characters": [], "script_blocks": [],
                    "semantic_annotations": {"scene_summary": None, "dramatic_function": None},
                    "parsing": {"status": "needs_review" if broken else "parsed", "needs_review": broken},
                }
                scenes.append(current)
                current_cue = None
                pending_parenthetical = None
                continue
            if current is None:
                continue
            current["script_pages"]["end"] = page_no
            if broken:
                current["parsing"] = {"status": "needs_review", "needs_review": True}
            width = float(page.get("width", 612))
            if _looks_like_cue(text, float(entry.get("x0", 0)), width):
                original = text
                speaker = normalize_character_cue(text)
                current_cue = (speaker, original)
                pending_parenthetical = None
                if speaker not in current["scene_characters"]:
                    current["scene_characters"].append(speaker)
                continue
            if current_cue and (pending_parenthetical is not None or text.startswith("(")):
                pending_parenthetical = f"{pending_parenthetical or ''} {text}".strip()
                if ")" not in pending_parenthetical:
                    continue
                parenthetical, _, remainder = pending_parenthetical.partition(")")
                pending_parenthetical = parenthetical + ")"
                text = remainder.strip()
                if not text:
                    continue
            # Dialogue is indented relative to action. Keep the cue active for
            # wrapped lines; an action-indented line ends the dialogue block.
            if current_cue and float(entry.get("x0", 0)) < width * 0.22:
                current_cue = None
                pending_parenthetical = None
            block_type = "dialogue" if current_cue else "action"
            source_block = entry.get("source_block")
            previous = current["script_blocks"][-1] if current["script_blocks"] else None
            if (
                previous is not None and previous["block_type"] == block_type
                and source_block is not None and previous.get("_source_block") == source_block
                and (block_type == "action" or previous.get("speaker") == current_cue[0])
            ):
                previous["text"] += " " + text
                continue
            number = 1 + sum(block["block_type"] == block_type for block in current["script_blocks"])
            block = {
                "block_id": f"{current['scene_id']}_{block_type}_{number:03d}",
                "block_type": block_type, "script_page": page_no,
                "source_order": len(current["script_blocks"]) + 1, "text": text,
                "_source_block": source_block,
            }
            if current_cue:
                block.update({"speaker": current_cue[0], "character_cue": current_cue[1], "parenthetical": pending_parenthetical})
                pending_parenthetical = None
            current["script_blocks"].append(block)

    for scene in scenes:
        for block in scene["script_blocks"]:
            block.pop("_source_block", None)

    return {
        "schema_version": "1.0",
        "movie": {"movie_id": movie_key, "title": title},
        "source_files": source_files,
        "summary": {
            "scene_count": len(scenes),
            "dialogue_block_count": sum(b["block_type"] == "dialogue" for s in scenes for b in s["script_blocks"]),
            "action_block_count": sum(b["block_type"] == "action" for s in scenes for b in s["script_blocks"]),
            "broken_page_count": len(broken_pages),
        },
        "broken_pages": broken_pages,
        "script_scenes": scenes,
    }


def extract_pdf_layout(path: Path) -> list[dict[str, Any]]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError('PyMuPDF is required; install with pip install -e ".[context]"') from exc
    pages: list[dict[str, Any]] = []
    with fitz.open(path) as document:
        for page_index, page in enumerate(document, 1):
            lines: list[dict[str, Any]] = []
            seen: set[tuple[str, float, float, float, float]] = set()
            raw = page.get_text("dict", sort=True)
            for block_number, block in enumerate(raw.get("blocks", [])):
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    text = "".join(str(span.get("text", "")) for span in spans).strip()
                    if text:
                        bbox = line.get("bbox", block.get("bbox", (0, 0, 0, 0)))
                        signature = (text, *(round(float(value), 1) for value in bbox))
                        if signature in seen:
                            continue
                        seen.add(signature)
                        lines.append({"text": text, "x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3], "source_block": block_number})
            pages.append({"page": page_index, "width": page.rect.width, "height": page.rect.height, "lines": lines})
    return pages


def parse_screenplay(path: Path, movie_key: str, title: str, source_files: dict[str, str]) -> dict[str, Any]:
    return parse_layout_pages(extract_pdf_layout(path), movie_key, title, source_files)
