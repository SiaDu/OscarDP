"""Prepare Stage06.1 inputs or publish an inspected VLM candidate."""

import argparse
from pathlib import Path

from oscardp.shots.frame_samples import load_json

from .inputs import create_sheets, inspected_images, prepare_inputs, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Build a shot manifest and labeled JPEG contact sheets")
    prepare.add_argument("--movie-dir", type=Path, required=True)
    prepare.add_argument("--movie-id", default="tt12300742")
    prepare.add_argument("--sequence-id", default="seq_006")
    prepare.add_argument("--output-dir", type=Path, required=True)

    render = commands.add_parser("render", help="Validate and render an existing visual candidate")
    render.add_argument("--prepared", type=Path, required=True)
    render.add_argument("--candidate", type=Path, required=True)
    render.add_argument("--inspected-sheet", action="append", required=True, metavar="JPG",
                        help="Sheets inspected for the candidate, including earlier visual evidence")
    augment = render.add_mutually_exclusive_group()
    augment.add_argument("--camera-groups", type=Path, help="New group array to add to a v1.0 candidate")
    augment.add_argument("--camera-clues", type=Path, help="Per-shot visual evidence to add to a v1.1 candidate")
    render.add_argument("--static-layout", metavar="LAYOUT_ID", help="Explicit opt-in to reuse a layout for this sequence")
    render.add_argument("--rechecked-sheet", action="append", metavar="JPG",
                        help="Sheets actually reviewed for the new visual inference")
    render.add_argument("--vlm", required=True, help="Actual VLM name and method used")
    render.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "render":
        if bool(args.camera_groups or args.camera_clues) != bool(args.rechecked_sheet):
            parser.error("incremental inference and --rechecked-sheet must be supplied together")
        if args.static_layout and not args.camera_clues:
            parser.error("--static-layout requires --camera-clues")

    try:
        if args.command == "prepare":
            packet = prepare_inputs(args.movie_dir, args.movie_id, args.sequence_id)
            args.output_dir.mkdir(parents=True, exist_ok=False)
            create_sheets(packet, args.output_dir)
            write_json(args.output_dir / "inputs.json", packet)
            count = sum(map(len, inspected_images(packet, packet["contact_sheets"]).values()))
            print(f"{len(packet['shots'])} shots, {count} source JPGs prepared: {args.output_dir}")
        else:
            from .output import make_review, make_svg
            from .validation import add_camera_groups, add_camera_regions, validate_candidate

            packet = load_json(args.prepared / "inputs.json")
            candidate = load_json(args.candidate)
            images = inspected_images(packet, args.inspected_sheet)
            rechecked = {}
            if args.rechecked_sheet:
                validate_candidate(candidate, packet, images)
                rechecked = inspected_images(packet, args.rechecked_sheet)
                if args.camera_groups:
                    candidate = add_camera_groups(candidate, load_json(args.camera_groups), rechecked)
                else:
                    if any(s["shot_id"] not in rechecked for s in candidate["shots"]):
                        raise ValueError("region inference requires rechecked images for every annotated shot")
                    candidate = add_camera_regions(candidate, load_json(args.camera_clues),
                                                   static_layout=args.static_layout)
                images.update(rechecked)
            validate_candidate(candidate, packet, images)
            svg = make_svg(candidate)
            review = make_review(candidate, packet, images, args, rechecked=rechecked)
            args.output_dir.mkdir(parents=True, exist_ok=False)
            write_json(args.output_dir / "scene_layout.json", candidate)
            for name, content in (("scene_layout.svg", svg), ("review.md", review)):
                with (args.output_dir / name).open("x", encoding="utf-8") as stream:
                    stream.write(content)
            print(f"Validated {len(candidate['shots'])} shots: {args.output_dir}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"stage06_1: {exc}\n")


if __name__ == "__main__":
    main()
