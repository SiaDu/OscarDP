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
    "SUNRISE", "SUNSET", "LATER", "CONTINUOUS", "MOMENTS LATER", "SAME TIME",
}
CUE_SUFFIX_RE = re.compile(r"\s*\((?:CONT['’]?D|CONTINUED|V\.?O\.?|O\.?S\.?|OFF)\)\s*$", re.I)
PAGE_NOISE_RE = re.compile(r"^(?:CONTINUED:?|\"?BLUE MOON\"?\s+CONFORMED SCRIPT.*|\d+[A-Z]?\.?)$", re.I)
MORE_MARKER_RE = re.compile(r"^\(?\s*MORE\s*\)?$", re.I)
SECTION_MARKER_RE = re.compile(r"^(?:PT|PART)\s+\d+\s*:$", re.I)
TRANSITION_RE = re.compile(
    r"^(?:==>\s*)?(?:CUT TO:?|FADE (?:IN|OUT):?|DISSOLVE(?: TO)?:?)(?:\s*==>)?$", re.I
)
CUE_FORMAT_LABELS = {
    "EMPHASIS", "NOTE", "INSERT", "FLASHBACK", "TITLE", "CARD", "CONTINUED",
    "FADE", "CUT", "DISSOLVE", "PAN", "ZOOM",
}
EDITORIAL_OBJECT_RE = re.compile(
    r"^THE\s+(?:PHOTOGRAPH|PHOTO|IMAGE|DOCUMENT)(?:\s+(?:IN\s+DETAIL|INSERT|CLOSE[- ]?UP))?\s*:?$",
    re.I,
)
NARRATIVE_SUBJECT = r"(?:He|She|They|[A-Z][a-zÀ-ÿ.-]+(?:/[A-Z][a-zÀ-ÿ.-]+)?(?:\s+(?:and|&)\s+[A-Z][a-zÀ-ÿ.-]+)*)"
NARRATIVE_ACTION_RE = re.compile(
    rf"^{NARRATIVE_SUBJECT}\s+(?:looks?|listens?|turns?|walks?|exits?|enters?|approaches?|remembers?|sees?|puts?|takes?|nods?|stares?|glances?|watches?|points?|sits?|stands?|moves?|opens?|closes?|reacts?|holds?|heads?|runs?|leaves?|steps?|continues?)\b",
    re.I,
)
NARRATIVE_APPOSITIVE_RE = re.compile(
    r"^[A-Z][A-ZÀ-Ÿ'’.-]+(?:/[A-Z][A-ZÀ-Ÿ'’.-]+)?,\s+[^.!?]+,\s+(?:adds?|says?|asks?|answers?|continues?|looks?|walks?|turns?)\b",
)
INLINE_OPEN_PAREN_RE = re.compile(r"^(?P<speech>.+?\S)\s+(?P<parenthetical>\([^()]*)$")


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
    ending_match = re.search(r"(?:\s*-\s*|[\.:]\s*)([^.:-]+?)\s*$", rest)
    if ending_match and (ending_match.group(1).upper() in TIME_WORDS or re.fullmatch(r"(?:19|20)\d{2}", ending_match.group(1))):
        return int_ext, rest[:ending_match.start()].strip(), ending_match.group(1).upper()
    return int_ext, rest, None


def _looks_like_cue(text: str, x0: float, width: float) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped) > 45 or stripped.startswith("("):
        return False
    if SCENE_RE.match(stripped) or EDITORIAL_OBJECT_RE.fullmatch(stripped) or stripped.endswith(('.', '!', '?', ':')):
        return False
    label_words = re.findall(r"[A-Z]+", stripped.upper())
    if label_words and label_words[0] in CUE_FORMAT_LABELS:
        return False
    letters = re.sub(r"[^A-Za-z]", "", stripped)
    return bool(letters) and stripped == stripped.upper() and x0 >= width * 0.36


def _looks_like_action_narrative(text: str, *, require_terminal: bool = True) -> bool:
    stripped = text.strip()
    terminal = stripped.rstrip("\"'’”)」]")
    if not terminal or (require_terminal and not terminal.endswith((".", "...", ":"))):
        return False
    first_word = stripped.split(maxsplit=1)[0]
    if first_word.casefold() == "it" or first_word.endswith((".", "!", "?", "-", "–", "—")):
        return False
    return bool(NARRATIVE_ACTION_RE.match(stripped) or NARRATIVE_APPOSITIVE_RE.match(stripped))


def _ends_active_cue_as_action(
    text: str, entry: dict[str, Any], previous: dict[str, Any] | None,
    current_cue: tuple[str, str] | None, width: float, page_no: int,
) -> bool:
    if current_cue is None or previous is None or previous.get("block_type") != "dialogue":
        return False
    if previous.get("speaker") != current_cue[0] or previous.get("_page") != page_no:
        return False
    source_block, previous_source = entry.get("source_block"), previous.get("_source_block")
    if not isinstance(source_block, int) or not isinstance(previous_source, int) or source_block == previous_source:
        return False
    x0, y0 = float(entry.get("x0", 0)), float(entry.get("y0", 0))
    vertical_gap = y0 - float(previous.get("_y1", y0))
    return width * 0.22 <= x0 <= width * 0.34 and 0 <= vertical_gap <= 72 and _looks_like_action_narrative(text, require_terminal=False)


def audit_screenplay_structure(context: dict[str, Any]) -> dict[str, Any]:
    affected: list[dict[str, Any]] = []
    action_like = editorial = fragmented = 0
    confirmed = 0
    for scene in context.get("script_scenes", []):
        for block in scene.get("script_blocks", []):
            if block.get("block_type") != "dialogue":
                continue
            reasons: list[str] = []
            speaker, text = str(block.get("speaker", "")), str(block.get("text", ""))
            parenthetical = block.get("parenthetical")
            if EDITORIAL_OBJECT_RE.fullmatch(speaker):
                editorial += 1
                reasons.append("editorial_label_speaker")
            if _looks_like_action_narrative(text) or (EDITORIAL_OBJECT_RE.fullmatch(speaker) and text.rstrip().endswith(":")):
                action_like += 1
                reasons.append("action_like_dialogue")
            if text.count("(") != text.count(")") or (
                isinstance(parenthetical, str) and parenthetical.count("(") != parenthetical.count(")")
            ):
                fragmented += 1
                reasons.append("fragmented_parenthetical")
            if reasons:
                is_confirmed = any(reason != "action_like_dialogue" for reason in reasons)
                confirmed += int(is_confirmed)
                affected.append({
                    "scene_id": scene.get("scene_id"), "block_id": block.get("block_id"),
                    "speaker": block.get("speaker"), "text": text,
                    "parenthetical": parenthetical, "reasons": reasons,
                    "confirmed_structural_error": is_confirmed,
                })
    return {
        "schema_version": "1.1", "action_like_dialogue_count": action_like,
        "editorial_label_speaker_count": editorial,
        "fragmented_parenthetical_count": fragmented,
        "confirmed_structural_error_count": confirmed,
        "diagnostic_only_count": len(affected) - confirmed,
        "affected_blocks": affected,
    }


def _wrapped_dialogue_continuation(
    text: str, entry: dict[str, Any], previous: dict[str, Any] | None,
    current_cue: tuple[str, str] | None, width: float, page_no: int,
) -> bool:
    if previous is None or current_cue is None or previous.get("block_type") != "dialogue":
        return False
    if previous.get("speaker") != current_cue[0] or previous.get("_page") != page_no:
        return False
    source_block, previous_source = entry.get("source_block"), previous.get("_source_block")
    if not isinstance(source_block, int) or not isinstance(previous_source, int) or source_block != previous_source + 1:
        return False
    x0, y0 = float(entry.get("x0", 0)), float(entry.get("y0", 0))
    displacement = float(previous.get("_x0", x0)) - x0
    vertical_gap = y0 - float(previous.get("_y1", y0))
    words = text.split()
    fragment_like = len(words) == 1 or (len(words) <= 3 and text[:1].islower())
    previous_open = not previous.get("text", "").rstrip().endswith((".", "!", "?", ";", ":"))
    return (
        previous_open and fragment_like and 0 <= vertical_gap <= 30
        and 0 <= displacement <= width * 0.14 and x0 >= width * 0.16
    )


def parse_layout_pages(pages: list[dict[str, Any]], movie_key: str, title: str, source_files: dict[str, str]) -> dict[str, Any]:
    scenes: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    pending_parenthetical: str | None = None
    pending_parenthetical_target: dict[str, Any] | None = None
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
            if MORE_MARKER_RE.fullmatch(text):
                continue
            if SECTION_MARKER_RE.fullmatch(text):
                current_cue = None
                pending_parenthetical = None
                pending_parenthetical_target = None
                continue
            heading = SCENE_RE.match(text)
            if heading:
                lead = (heading.group("lead") or "").upper()
                trail = (heading.group("trail") or "").upper()
                trailing_year = trail if re.fullmatch(r"(?:19|20)\d{2}", trail) else ""
                raw_no = lead or ("" if trailing_year else trail)
                if not raw_no:
                    same_line = [value for y, value in numeric_by_y if abs(y - float(entry.get("y0", 0))) <= 2.5]
                    if same_line:
                        raw_no = same_line[0]
                slugline = heading.group("slug").strip()
                if trailing_year:
                    slugline += f" {trailing_year}"
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
                pending_parenthetical_target = None
                continue
            if current is None:
                continue
            current["script_pages"]["end"] = page_no
            if broken:
                current["parsing"] = {"status": "needs_review", "needs_review": True}
            upper_text = text.rstrip(".").upper()
            if (
                upper_text in TIME_WORDS and not current["script_blocks"]
                and current["script_pages"]["start"] == page_no
            ):
                current["slugline"] = current["slugline"].rstrip(".") + f". {upper_text}"
                current["int_ext"], current["location"], current["time_of_day"] = _split_slugline(current["slugline"])
                current_cue = None
                pending_parenthetical = None
                pending_parenthetical_target = None
                continue
            is_transition = bool(TRANSITION_RE.fullmatch(text))
            if is_transition:
                current_cue = None
                pending_parenthetical = None
                pending_parenthetical_target = None
            width = float(page.get("width", 612))
            if not is_transition and _looks_like_cue(text, float(entry.get("x0", 0)), width):
                original = text
                speaker = normalize_character_cue(text)
                current_cue = (speaker, original)
                pending_parenthetical = None
                pending_parenthetical_target = None
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
                if pending_parenthetical_target is not None:
                    pending_parenthetical_target["parenthetical"] = pending_parenthetical
                    pending_parenthetical = None
                    pending_parenthetical_target = None
                if not text:
                    continue
            previous = current["script_blocks"][-1] if current["script_blocks"] else None
            if _ends_active_cue_as_action(text, entry, previous, current_cue, width, page_no):
                if pending_parenthetical is not None and previous is not None:
                    previous_parenthetical = previous.get("parenthetical")
                    previous["parenthetical"] = (
                        f"{previous_parenthetical} {pending_parenthetical}" if previous_parenthetical else pending_parenthetical
                    )
                current_cue = None
                pending_parenthetical = None
                pending_parenthetical_target = None
            wrapped_continuation = _wrapped_dialogue_continuation(text, entry, previous, current_cue, width, page_no)
            if wrapped_continuation:
                previous["text"] += " " + text
                previous["_source_block"] = entry.get("source_block")
                previous["_x0"] = float(entry.get("x0", 0))
                previous["_y1"] = float(entry.get("y1", entry.get("y0", 0)))
                continue
            # Dialogue is indented relative to action. Keep the cue active for
            # wrapped lines; an action-indented line ends the dialogue block.
            if current_cue and float(entry.get("x0", 0)) < width * 0.22:
                current_cue = None
                pending_parenthetical = None
                pending_parenthetical_target = None
            inline_parenthetical: str | None = None
            if current_cue:
                inline_match = INLINE_OPEN_PAREN_RE.match(text)
                if inline_match:
                    text = inline_match.group("speech").strip()
                    inline_parenthetical = inline_match.group("parenthetical").strip()
            block_type = "dialogue" if current_cue else "action"
            source_block = entry.get("source_block")
            if (
                previous is not None and previous["block_type"] == block_type
                and source_block is not None and previous.get("_source_block") == source_block
                and (block_type == "action" or previous.get("speaker") == current_cue[0])
            ):
                previous["text"] += " " + text
                previous["_x0"] = float(entry.get("x0", previous.get("_x0", 0)))
                previous["_y1"] = float(entry.get("y1", entry.get("y0", previous.get("_y1", 0))))
                if inline_parenthetical is not None:
                    pending_parenthetical = inline_parenthetical
                    pending_parenthetical_target = previous
                continue
            number = 1 + sum(block["block_type"] == block_type for block in current["script_blocks"])
            block = {
                "block_id": f"{current['scene_id']}_{block_type}_{number:03d}",
                "block_type": block_type, "script_page": page_no,
                "source_order": len(current["script_blocks"]) + 1, "text": text,
                "_source_block": source_block, "_x0": float(entry.get("x0", 0)),
                "_y1": float(entry.get("y1", entry.get("y0", 0))), "_page": page_no,
            }
            if current_cue:
                block.update({"speaker": current_cue[0], "character_cue": current_cue[1], "parenthetical": pending_parenthetical})
                pending_parenthetical = None
            current["script_blocks"].append(block)
            if inline_parenthetical is not None:
                pending_parenthetical = inline_parenthetical
                pending_parenthetical_target = block

    for scene in scenes:
        for block in scene["script_blocks"]:
            for private in ("_source_block", "_x0", "_y1", "_page"):
                block.pop(private, None)

    result = {
        "schema_version": "1.0",
        "parser_version": "2.4.1",
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
    result["parsing_audit"] = audit_screenplay_structure(result)
    return result


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
