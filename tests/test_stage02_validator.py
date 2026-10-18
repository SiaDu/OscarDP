from copy import deepcopy
import json
from pathlib import Path

import pytest

from oscardp.stage02.__main__ import main
from oscardp.stage02.srt import compare_text, index_srts, parse_srt
from oscardp.stage02.validator import SCHEMA, validate_movie

ID = 'tt1234567'
SRT = '1\n00:00:00,500 --> 00:00:01,500\nHello.\n\n2\n00:00:01,500 --> 00:00:02,500\nGoodbye.\n'


def make_movie(root, movie_id=ID):
    directory = root / movie_id
    directory.mkdir(parents=True)
    shots = [dict(shot_id=f'shot_{i + 1:06d}', temporal=dict(start=i, end=i + 1, start_frame=i * 24, end_frame=(i + 1) * 24)) for i in range(3)]
    one = dict(movie_id=movie_id, movie_name='Test Movie', shots=shots)
    two = deepcopy(one)
    for shot in two['shots']:
        shot['temporal'] = {k: v for k, v in shot['temporal'].items() if k in ('start', 'end')}
        shot.update(dialogue=[], narrative=dict(scene='Test', location=None, time_of_day='Unknown', interior_exterior='Unknown', characters_in_script_context=[], action=[]), tags=[])
    for stage, data in [('Stage01', one), ('Stage02', two)]:
        (directory / f'{stage}_{movie_id}.json').write_text(json.dumps(data))
    return directory, one, two


@pytest.fixture
def movie(tmp_path):
    directory, one, two = make_movie(tmp_path / 'analysis')
    srt = tmp_path / f'{ID}.srt'
    srt.write_text(SRT)
    return directory, one, two, srt


def run(movie, mutate=None):
    directory, _, two, srt = movie
    two = deepcopy(two)
    if mutate:
        mutate(two)
    (directory / f'Stage02_{ID}.json').write_text(json.dumps(two))
    return validate_movie(directory, [srt])


def codes(report):
    return {i['rule_code'] for i in report['issues']}


def dialogue(indices=(1,), text='Hello.', speaker=None):
    return dict(srt_indices=list(indices), text=text, speaker=speaker)


def test_valid_empty_optional_and_null(movie):
    report = run(movie)
    assert report['summary']['validation_status'] == 'PASS'
    assert not report['incomplete_checks']
    report = run(movie, lambda d: d['shots'][0]['dialogue'].append(dialogue()))
    assert report['summary']['validation_status'] == 'PASS'
    assert report['summary']['unknown_speaker_count'] == 1


@pytest.mark.parametrize('case,code', [('missing', 'SHOT_MISSING'), ('duplicate', 'SHOT_DUPLICATE'), ('extra', 'SHOT_EXTRA'), ('order', 'SHOT_ORDER')])
def test_coverage(movie, case, code):
    def mutate(d):
        if case == 'missing':
            d['shots'].pop(0)
        elif case == 'duplicate':
            d['shots'].append(deepcopy(d['shots'][0]))
        elif case == 'extra':
            d['shots'][0]['shot_id'] = 'shot_999999'
        else:
            d['shots'].reverse()
    report = run(movie, mutate)
    assert code in codes(report)
    assert report['summary']['validation_status'] == 'FAIL'


@pytest.mark.parametrize('delta,failed', [(0.0000001, False), (0.00001, True)])
def test_tolerance(movie, delta, failed):
    report = run(movie, lambda d: d['shots'][0]['temporal'].update(end=1 + delta))
    assert ('TIME_MISMATCH' in codes(report)) == failed
    assert report['tolerance_seconds'] == 1e-6


@pytest.mark.parametrize('mutate', [
    lambda d: d.update(unassigned_subtitle_cues=[]),
    lambda d: d['shots'][0].update(page=1),
    lambda d: d['shots'][0]['narrative'].pop('scene'),
    lambda d: d['shots'][0]['narrative'].update(action=['invented']),
    lambda d: d['shots'][0]['narrative'].update(characters_in_script_context='A'),
    lambda d: d['shots'][0].update(dialogue=[dialogue(indices=(True,))]),
    lambda d: d['shots'][0].update(dialogue=[dialogue(indices=(1.0,))]),
    lambda d: d['shots'][0].update(dialogue=[dialogue(indices=())]),
])
def test_schema(movie, mutate):
    assert 'SCHEMA_INVALID' in codes(run(movie, mutate))


def test_cross_shot_and_merge(movie):
    report = run(movie, lambda d: [d['shots'][i]['dialogue'].append(dialogue()) for i in (0, 1)])
    assert report['summary']['validation_status'] == 'PASS'
    report = run(movie, lambda d: d['shots'][1]['dialogue'].append(dialogue((1, 2), 'Hello.\nGoodbye.')))
    assert report['summary']['exact_match_count'] == 1
    assert report['summary']['validation_status'] == 'PASS'


@pytest.mark.parametrize('indices', [(2, 1), (1, 1)])
def test_invalid_merge(movie, indices):
    report = run(movie, lambda d: d['shots'][1]['dialogue'].append(dialogue(indices, 'Goodbye. Hello.')))
    assert 'SRT_MERGE_ORDER' in codes(report)


def test_missing_source_and_no_overlap(movie):
    report = run(movie, lambda d: d['shots'][0]['dialogue'].append(dialogue((99,))))
    assert 'SRT_INDEX_MISSING' in codes(report)
    report = run(movie, lambda d: d['shots'][2]['dialogue'].append(dialogue()))
    assert 'SRT_NO_OVERLAP' in codes(report)


def test_overlap_uses_stage01(movie):
    def mutate(d):
        d['shots'][2]['temporal'] = dict(start=0, end=1)
        d['shots'][2]['dialogue'] = [dialogue()]
    assert {'TIME_MISMATCH', 'SRT_NO_OVERLAP'} <= codes(run(movie, mutate))


@pytest.mark.parametrize('text,status', [('Hello.', 'PASS'), (' <i>Hello.</i>\n', 'WARNING'), ('hello.', 'FAIL'), ('Hello!', 'FAIL')])
def test_text_levels(movie, text, status):
    report = run(movie, lambda d: d['shots'][0]['dialogue'].append(dialogue(text=text)))
    assert report['summary']['validation_status'] == status
    assert report['text_comparisons'][0]['source_cues'][0]['text'] == 'Hello.'


def test_bom_crlf_markup_unicode():
    parsed = parse_srt('\ufeff1\r\n00:00:00,100 --> 00:00:01,200\r\n<i>Café</i>\r\n&amp; tea\r\n')
    assert not parsed.issues
    assert parsed.cues[0].text == '<i>Café</i>\n&amp; tea'
    result = compare_text('Cafe\u0301 & tea', parsed.cues)
    assert result['status'] == 'normalized'
    assert set(result['reasons']) == {'subtitle_markup', 'html_entities', 'unicode_nfc', 'whitespace'}


def test_alignment_is_formatting_only():
    cue = parse_srt('1\n00:00:00,100 --> 00:00:01,200\n{\\an8}<i>Hello.</i>').cues
    assert compare_text('Hello.', cue)['status'] == 'normalized'
    assert compare_text('{unknown}Hello.', cue)['status'] == 'mismatch'


def test_merge_preserves_original_nonsequential_numbers(movie):
    directory, _, _, srt = movie
    srt.write_text(SRT.replace('\n2\n', '\n5\n'))
    report = run(movie, lambda d: d['shots'][1]['dialogue'].append(dialogue((1, 5), 'Hello.\nGoodbye.')))
    assert report['summary']['validation_status'] == 'PASS'


def test_webvtt_is_not_numbered_srt(movie):
    directory, _, _, srt = movie
    srt.write_text('WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello.')
    report = run(movie, lambda d: d['shots'][0]['dialogue'].append(dialogue()))
    assert 'SRT_UNSUPPORTED_FORMAT' in codes(report)
    assert 'SRT_INDEX_MISSING' not in codes(report)
    assert 'source_indices' in report['incomplete_checks']


@pytest.mark.parametrize('text,code', [
    (SRT + '\n' + SRT, 'SRT_DUPLICATE_INDEX'),
    ('1\n00:60:00,000 --> 00:00:01,000\nText', 'SRT_TIMESTAMP'),
    ('1\n00:00:02,000 --> 00:00:01,000\nText', 'SRT_TIMESTAMP'),
    ('foo\nbar', 'SRT_FORMAT'),
    ('', 'SRT_EMPTY'),
    ('1\n00:00:00,000 --> 00:00:01,000\nText\n2\n00:00:01,000 --> 00:00:02,000\nOther', 'SRT_FORMAT'),
])
def test_srt_errors(text, code):
    assert code in {i['rule_code'] for i in parse_srt(text).issues}


@pytest.mark.parametrize('basis,verification,valid', [
    ('screenplay_explicit', 'screenplay_text_only_not_video_observation', True),
    ('llm_inferred', 'screenplay_text_only_not_video_observation', True),
    ('video_observed', 'screenplay_text_only_not_video_observation', False),
    ('llm_inferred', 'video_verified', False),
])
def test_performance(movie, basis, verification, valid):
    cue = dict(character='A', category='gaze_aversion', description='Text evidence', basis=basis, verification=verification)
    report = run(movie, lambda d: d['shots'][0].update(performance_cues=[cue]))
    assert ('SCHEMA_INVALID' not in codes(report)) == valid


@pytest.mark.parametrize('field', ['sound_cues', 'song_lyrics'])
def test_optional_source_fields(movie, field):
    report = run(movie, lambda d: d['shots'][0].update({field: [dict(srt_index=1, text='Hello.')]}))
    assert report['summary']['validation_status'] == 'PASS'
    report = run(movie, lambda d: d['shots'][0].update({field: [dict(srt_index=2, text='Wrong')]}))
    assert {'TEXT_MISMATCH', 'SRT_NO_OVERLAP'} <= codes(report)


def test_discovery_conflict_and_override(movie, tmp_path):
    directory, _, _, srt = movie
    other = tmp_path / 'other.srt'
    other.write_text(SRT)
    assert 'SRT_AMBIGUOUS' in codes(validate_movie(directory, [srt, other]))
    assert validate_movie(directory, [srt, other], srt_override=srt)['summary']['validation_status'] == 'PASS'
    missing = validate_movie(directory, [])
    assert missing['summary']['validation_status'] == 'FAIL'
    assert 'source_text' in missing['incomplete_checks']


def test_index_filename_parent_same_directory_and_no_titles(tmp_path):
    for name in ['tt1234567.srt', 'Movie Name.srt', 'folder tt7654321/captions.srt', 'video_folder/subtitles.srt']:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(SRT)
    (tmp_path / 'video_folder/tt0000001.mkv').touch()
    index = index_srts(tmp_path)
    assert set(index) == {ID, 'tt7654321', 'tt0000001'}
    (tmp_path / 'video_folder/tt0000002.mkv').touch()
    assert index_srts(tmp_path).ambiguous_ids == {'tt0000001', 'tt0000002'}


def test_batch_18_continue_and_reports(tmp_path, capsys):
    analysis, srts, reports = (tmp_path / name for name in ('analysis', 'srts', 'reports'))
    srts.mkdir()
    before = {}
    for i in range(18):
        movie_id = f'tt{i + 1:07d}'
        directory, _, _ = make_movie(analysis, movie_id)
        (srts / f'{movie_id}.srt').write_text(SRT)
        if i == 0:
            (directory / f'Stage02_{movie_id}.json').write_text('{broken')
    for root in (analysis, srts):
        before.update({p: p.read_bytes() for p in root.rglob('*') if p.is_file()})
    argv = ['validate', '--analysis-root', str(analysis), '--srt-root', str(srts), '--report-root', str(reports), '--expect-movies', '18']
    assert main(argv) == 1
    summary = json.loads((reports / 'summary.json').read_text())
    assert summary['counts'] == dict(PASS=17, WARNING=0, FAIL=1)
    assert len(list((reports / 'movies').glob('*.json'))) == 18
    assert '[18/18]' in capsys.readouterr().out
    assert all(p.read_bytes() == content for p, content in before.items())
    assert main(argv) == 2  # Existing reports never overwritten.


def test_count_mismatch_and_empty_batch(tmp_path):
    for name in ('analysis', 'srts'):
        (tmp_path / name).mkdir()
    args = ['validate', '--analysis-root', str(tmp_path / 'analysis'), '--srt-root', str(tmp_path / 'srts'), '--report-root', str(tmp_path / 'reports'), '--expect-movies', '18']
    assert main(args) == 1
    summary = json.loads((tmp_path / 'reports/summary.json').read_text())
    assert {i['rule_code'] for i in summary['batch_issues']} == {'NO_MOVIES', 'MOVIE_COUNT'}


def test_partial_is_not_complete(movie):
    directory, _, _, srt = movie
    (directory / f'Stage02_{ID}.json').rename(directory / f'Stage02_{ID}_part001.json')
    assert 'INPUT_MISSING' in codes(validate_movie(directory, [srt]))


def test_schema_distribution():
    root = Path(__file__).resolve().parents[1]
    assert json.loads((root / 'docs/schemas/stage02_v1.schema.json').read_text()) == SCHEMA


@pytest.mark.parametrize('bad', ['null', '{"movie_id": 1, "movie_id": 2}', '{"x": NaN}', '{"x": 1e999}'])
def test_bad_json(movie, bad):
    directory, _, _, srt = movie
    (directory / f'Stage02_{ID}.json').write_text(bad)
    assert validate_movie(directory, [srt])['summary']['validation_status'] == 'FAIL'


def test_single_cli_with_mapping(movie, tmp_path):
    directory, _, _, srt = movie
    mapping = tmp_path / 'map.json'
    mapping.write_text(json.dumps({ID: srt.name}))
    assert main(['validate', '--analysis-root', str(directory.parent), '--srt-root', str(tmp_path / 'unavailable'), '--report-root', str(tmp_path / 'reports'), '--movie-id', ID, '--srt-map', str(mapping)]) == 0


def test_reject_reports_in_inputs(movie):
    directory, _, _, _ = movie
    assert main(['validate', '--analysis-root', str(directory.parent), '--report-root', str(directory / 'reports')]) == 2
