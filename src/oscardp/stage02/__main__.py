"""CLI: python -m oscardp.stage02 validate --help."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

from .reporting import write_reports
from .srt import index_srts
from .validator import MovieValidation, finite, read_json, validate_movie

MOVIE_ID = re.compile(r'tt[0-9]{7,10}')
DEFAULT_ANALYSIS = Path('/media/sdu/SiyaoDu/datasets/oscar_movie_analysis')
DEFAULT_SRT = Path('/media/sdu/SiyaoDu/datasets/oscar_movie')


def load_srt_map(path: Path | None) -> dict[str, Path]:
    if path is None:
        return {}
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError('SRT map must be a JSON object: {"tt1234567": "/path/to/source.srt"}')
    result = {}
    for movie_id, value in data.items():
        if not MOVIE_ID.fullmatch(movie_id) or not isinstance(value, str) or not value.strip():
            raise ValueError(f'Invalid SRT mapping: {movie_id!r}')
        source = Path(value).expanduser()
        result[movie_id] = (source if source.is_absolute() else path.resolve().parent / source).resolve()
    return result


def run(args) -> int:
    if not finite(args.tolerance) or args.tolerance < 0:
        raise ValueError('--tolerance must be finite and nonnegative')
    if args.expect_movies is not None and args.expect_movies < 0:
        raise ValueError('--expect-movies must be nonnegative')
    if args.movie_id and not MOVIE_ID.fullmatch(args.movie_id):
        raise ValueError('--movie-id must be a canonical IMDb ID')
    analysis, srts, output = (p.expanduser().resolve() for p in (args.analysis_root, args.srt_root, args.report_root))
    for source_root in (analysis, srts):
        if output == source_root or output.is_relative_to(source_root):
            raise ValueError('--report-root must be outside the input dataset roots')
    if output.is_symlink() or (output.exists() and any(output.iterdir())):
        raise FileExistsError('Report directory is not empty; select a new --report-root')
    mapping = load_srt_map(args.srt_map)
    if not analysis.is_dir():
        raise FileNotFoundError(f'Analysis root does not exist: {analysis}')
    dirs = [analysis / args.movie_id] if args.movie_id else sorted(p for p in analysis.iterdir() if p.is_dir() and not p.is_symlink() and MOVIE_ID.fullmatch(p.name))
    batch_issues = []
    def batch_error(code, message, expected=None, actual=None):
        batch_issues.append(dict(movie_id=None, severity='ERROR', rule_code=code, shot_id=None, srt_index=None, expected=expected, actual=actual, message=message))
    if not dirs:
        batch_error('NO_MOVIES', 'No IMDb movie directories discovered')
    if args.expect_movies is not None and len(dirs) != args.expect_movies:
        batch_error('MOVIE_COUNT', 'Discovered movie count differs from expectation', args.expect_movies, len(dirs))
    discovery_error = None
    try:
        index = index_srts(srts)
    except OSError as exc:
        index = {}
        discovery_error = str(exc)
    reports = []
    for number, directory in enumerate(dirs, 1):
        try:
            report = validate_movie(directory, index.get(directory.name, []), srt_override=mapping.get(directory.name),
                                    tolerance=args.tolerance, discovery_error=discovery_error,
                                    ambiguous_ownership=directory.name in getattr(index, 'ambiguous_ids', set()))
        except Exception as exc:
            # Unexpected malformed input must never prevent validation of later movies.
            validation = MovieValidation(directory, args.tolerance)
            validation.issue('files_identity', 'VALIDATION_EXCEPTION', f'{type(exc).__name__}: {exc}')
            validation.skip(*validation.results)
            report = validation.finish()
        reports.append(report)
        stat = report['summary']
        print(f'[{number:02d}/{len(dirs):02d}] {directory.name}  {stat["validation_status"]} ({stat["errors"]} errors, {stat["warnings"]} warnings)', flush=True)
    summary = write_reports(output, reports, batch_issues, tolerance=args.tolerance)
    print(f'\nValidation completed.\nMovies: {len(dirs)}')
    for status, count in summary['counts'].items():
        print(f'{status}: {count}')
    for issue in batch_issues:
        print(f'{issue["rule_code"]}: {issue["message"]}', file=sys.stderr)
    print(f'Reports: {output}\nStructural and Source Consistency Validation; no semantic or video verification.')
    return 1 if summary['validation_status'] == 'FAIL' else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Stage02 structural and source consistency validation (read-only inputs)')
    subparsers = parser.add_subparsers(dest='command', required=True)
    validate = subparsers.add_parser('validate', help='Validate complete Stage02 results against Stage01 and original SRT')
    validate.add_argument('--analysis-root', type=Path, default=DEFAULT_ANALYSIS)
    validate.add_argument('--srt-root', type=Path, default=DEFAULT_SRT)
    validate.add_argument('--report-root', type=Path, default=Path('reports/stage02_validation'))
    validate.add_argument('--movie-id')
    validate.add_argument('--expect-movies', type=int)
    validate.add_argument('--srt-map', type=Path, help='Explicit IMDb-to-SRT JSON map; relative paths resolve against map directory')
    validate.add_argument('--tolerance', type=float, default=1e-6, help='Shot boundary serialization tolerance in seconds (default: 1e-6)')
    review = subparsers.add_parser('review', help='Reconcile saved runs and propose source/annotation corrections without applying them')
    review.add_argument('--baseline-root', type=Path, required=True)
    review.add_argument('--validation-root', type=Path, required=True)
    review.add_argument('--report-root', type=Path, required=True)
    review.add_argument('--historical-baseline', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'review':
            from .review import run_review
            audit = run_review(args.baseline_root, args.validation_root, args.report_root, historical=args.historical_baseline)
            print(f'Reviewed {audit["reviewed_issue_occurrences"]} baseline issues; {audit["independent_baseline_roots"]} independent roots.\nReview: {args.report_root / "REVIEW.md"}\nAll source and annotation proposals remain unapplied.')
            return 0
        return run(args)
    except (OSError, ValueError) as exc:
        print(f'Validation could not complete: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
