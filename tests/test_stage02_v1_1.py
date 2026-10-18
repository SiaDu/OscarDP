"""Regression tests for bounded mixed cues and source/annotation separation."""
import json
from copy import deepcopy

import pytest

from oscardp.stage02.srt import Cue, parse_srt
from oscardp.stage02.text import compare_entry, segments
from oscardp.stage02.validator import validate_movie
from test_stage02_validator import ID, SRT, make_movie


@pytest.mark.parametrize('field,text', [
    ('dialogue', 'Yeah, Coach, yeah!'),
    ('song_lyrics', "♪ Everybody's gotta learn\nsometime ♪"),
])
def test_mixed_music_and_dialogue(field, text):
    cue = Cue(739, 1, 3, "-♪ Everybody's gotta learn\nsometime ♪ -Yeah, Coach, yeah!")
    match = compare_entry(text, [cue], field)
    assert match['status'] == 'exact'
    assert match['match_rule'] == 'complete_typed_segment'
    assert not match['classification_error']
    assert match['extraction']['parts']


@pytest.mark.parametrize('text', ['Coach', 'Yeah, Coach', 'learn', 'sometime'])
def test_reject_arbitrary_substrings(text):
    cue = Cue(1, 0, 1, '♪ learn sometime ♪ -Yeah, Coach, yeah!')
    assert compare_entry(text, [cue], 'dialogue')['status'] == 'mismatch'


def test_plain_dialogue_line_cannot_be_sliced():
    cue = Cue(1, 0, 1, 'Hello there.\nGoodbye.')
    assert compare_entry('Hello there.', [cue], 'dialogue')['status'] == 'mismatch'


@pytest.mark.parametrize('field,text', [('sound_cues', '(sniffles)'), ('dialogue', 'Ooh.\nMAN: All right.')])
def test_complete_sound_segment_and_projection(field, text):
    cue = Cue(57, 0, 1, '-Ooh. (sniffles)\n-MAN: All right.')
    match = compare_entry(text, [cue], field)
    assert match['status'] == 'exact'
    assert match['match_rule'] in ('complete_typed_segment', 'complete_field_projection')


def test_full_mixed_lyric_entry_is_classification_error():
    cue = Cue(1, 0, 1, '♪ Song ♪ -Actual dialogue.')
    match = compare_entry(cue.text, [cue], 'song_lyrics')
    assert match['status'] == 'exact'  # Text provenance alone is insufficient.
    assert match['classification_error']
    assert compare_entry('Actual dialogue.', [cue], 'song_lyrics')['classification_error']


@pytest.mark.parametrize('source', ['[all] <i>♪ Song ♪</i>', '(singing along): ♪ Song ♪'])
def test_lyric_speaker_metadata_is_not_spoken_dialogue(source):
    assert not compare_entry(source, [Cue(1, 0, 1, source)], 'song_lyrics')['classification_error']


def test_instrumental_notes_are_sound_not_lyric_words():
    assert segments(Cue(1, 0, 1, '♪ ♪'))['segments'][0]['kind'] == 'sound_cues'
    assert not compare_entry('♪ ♪', [Cue(1, 0, 1, '♪ ♪')], 'sound_cues')['classification_error']


def test_lyric_then_dialogue_with_source_bracket_label():
    cue = Cue(1, 0, 1, "<i>- ♪ Lyrics ♪</i>\n- [John] <i>Hey, Bob.</i>")
    assert compare_entry('Hey, Bob.', [cue], 'dialogue')['status'] == 'exact'
    assert compare_entry('Hey', [cue], 'dialogue')['status'] == 'mismatch'


def test_turn_before_lyric_does_not_pollute_dialogue_segment():
    cue = Cue(1, 0, 1, '- very nice.\n<i>- ♪ A-wim-a-weh ♪</i>')
    assert compare_entry('very nice.', [cue], 'dialogue')['status'] == 'exact'


def fixture_movie(tmp_path, source, entries):
    directory, _, two = make_movie(tmp_path / 'analysis')
    for shot in two['shots'][:2]:
        shot.update(deepcopy(entries))
    (directory / f'Stage02_{ID}.json').write_text(json.dumps(two))
    srt = tmp_path / f'{ID}.srt'
    srt.write_text(source)
    return validate_movie(directory, [srt])


def test_cross_shot_root_dedup_retains_occurrences(tmp_path):
    report = fixture_movie(tmp_path, SRT, dict(dialogue=[dict(srt_indices=[1], text='Changed text', speaker=None)]))
    errors = [i for i in report['issues'] if i['rule_code'] == 'TEXT_MISMATCH']
    assert len(errors) == 2
    assert len({i['root_problem_id'] for i in errors}) == 1
    root = next(r for r in report['root_problems'] if r['root_problem_id'] == errors[0]['root_problem_id'])
    assert root['affected_shots'] == ['shot_000001', 'shot_000002']
    assert report['summary']['root_errors'] == 1


def test_duplicate_source_disambiguated_without_hiding_quality_error(tmp_path):
    source = SRT + '\n1\n00:01:00,000 --> 00:01:10,000\nAdvertise here.\n'
    report = fixture_movie(tmp_path, source, dict(dialogue=[dict(srt_indices=[1], text='Hello.', speaker=None)]))
    assert report['summary']['source_integrity_status'] == 'FAIL'
    assert report['summary']['annotation_consistency_status'] == 'PASS'
    assert report['summary']['root_errors'] == 1
    assert 'SRT_DUPLICATE_REFERENCE_DISAMBIGUATED' in {i['rule_code'] for i in report['issues']}


@pytest.mark.parametrize('source', [
    '1\n00:00:00,500 --> 00:00:01,500\n',
    '1\n00:00:00,000 --> 00:00:00,000\nHello.',
])
def test_invalid_referenced_source_is_blocked_not_missing(source, tmp_path):
    report = fixture_movie(tmp_path, source, dict(dialogue=[dict(srt_indices=[1], text='Hello.', speaker=None)]))
    assert report['summary']['source_integrity_status'] == 'FAIL'
    assert report['summary']['annotation_consistency_status'] == 'INCOMPLETE'
    assert 'SRT_INDEX_MISSING' not in {i['rule_code'] for i in report['issues']}


def test_webvtt_cascade_suppression(tmp_path):
    report = fixture_movie(tmp_path, 'WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nHello.',
                           dict(dialogue=[dict(srt_indices=[1], text='Hello.', speaker=None)] * 100))
    assert report['summary']['errors'] == 1
    assert report['summary']['annotation_consistency_status'] == 'INCOMPLETE'
    assert report['summary']['source_integrity_status'] == 'FAIL'


def test_rejected_blocks_keep_exact_original_evidence():
    parsed = parse_srt('1\n00:00:00,000 --> 00:00:00,000\nHello.')
    assert parsed.blocks[0]['text'] == 'Hello.'
    assert parsed.blocks[0]['start'] == parsed.blocks[0]['end'] == 0
    assert not parsed.blocks[0]['time_valid']


def test_ass_and_literal_language_are_separate():
    cue = Cue(1, 0, 1, '{\\an8}<i>Hello!</i>')
    assert compare_entry('Hello!', [cue], 'dialogue')['status'] == 'normalized'
    assert compare_entry('Hello.', [cue], 'dialogue')['status'] == 'mismatch'


@pytest.mark.parametrize('source', ['MARK:\n♪ Song ♪', 'CLAIRE (singing): ♪ Song ♪', 'MIKE and CLAIRE:\n♪ Song ♪', '-♪ First ♪\n--♪ Second ♪'])
def test_caption_speaker_label_is_not_dialogue(source):
    assert not compare_entry(source, [Cue(1, 0, 1, source)], 'song_lyrics')['classification_error']


def test_negative_number_not_a_turn_delimiter():
    cue = Cue(1, 0, 1, '♪ Song ♪ -5 degrees.')
    assert compare_entry('5 degrees.', [cue], 'dialogue')['status'] == 'mismatch'


def test_review_reconciles_every_issue_and_only_proposes_patches(tmp_path):
    from oscardp.stage02.reporting import write_reports
    from oscardp.stage02.review import run_review
    directory, _, two = make_movie(tmp_path / 'analysis')
    cue = '-♪ Song ♪ -Hello.'
    for shot in two['shots'][:2]:
        shot['dialogue'] = [dict(srt_indices=[1], speaker=None, text='Hello.')]
        shot['song_lyrics'] = [dict(srt_index=1, text=cue)]
    target = directory / f'Stage02_{ID}.json'
    target.write_text(json.dumps(two))
    srt = tmp_path / 'srts' / f'{ID}.srt'
    srt.parent.mkdir()
    srt.write_text('1\n00:00:00,500 --> 00:00:01,500\n' + cue)
    current = validate_movie(directory, [srt])
    before = {p: p.read_bytes() for p in [target, srt]}
    old = deepcopy(current)
    old['issues'] = [dict(movie_id=ID, severity='ERROR', rule_code='TEXT_MISMATCH', shot_id=f'shot_{i + 1:06d}',
                          srt_index=[1], expected=cue, actual='Hello.', message='Old whole-cue mismatch') for i in range(2)]
    for comparison in old['text_comparisons']:
        if comparison['field'] == 'dialogue':
            comparison['status'] = 'mismatch'
    baseline, validation, output = [tmp_path / name for name in ('baseline', 'validation', 'review')]
    write_reports(baseline, [old], [], tolerance=1e-6)
    write_reports(validation, [current], [], tolerance=1e-6)
    audit = run_review(baseline, validation, output)
    assert audit['reviewed_issue_occurrences'] == 2
    assert audit['independent_baseline_roots'] == 1
    assert audit['decision_counts']['RESOLVED_VALIDATOR_RULE'] == 2
    patches = json.loads((output / 'proposed_stage02_patches.json').read_text())['proposals']
    assert len(patches) == 2
    assert all(p['before'] == cue and p['after'] == '♪ Song ♪' and not p['applied'] for p in patches)
    assert all(p.read_bytes() == value for p, value in before.items())
    assert json.loads((output / 'verification.json').read_text())['all_hashes_unchanged']
    with pytest.raises(FileExistsError):
        run_review(baseline, validation, output)


def test_curation_proposal_keeps_original_and_number_map(tmp_path):
    from oscardp.stage02.curation import investigate
    directory, _, _ = make_movie(tmp_path / 'analysis')
    srts = tmp_path / 'srts'
    srts.mkdir()
    original = srts / f'{ID}.srt'
    original.write_text(SRT + '\n2\n00:01:00,000 --> 00:01:10,000\nAdvertise your product or brand here\n')
    before = original.read_bytes()
    result = investigate(validate_movie(directory, [original]), tmp_path / 'review')
    proposal = result['proposals'][0]
    assert original.read_bytes() == before
    assert not proposal['applied_to_dataset']
    assert proposal['candidate_validation']['source_integrity_status'] == 'PASS'
    assert [m['derived_index'] for m in proposal['original_index_mapping']] == [1, 2, None]
    assert proposal['requires_human_confirmation']


def test_zero_duration_nonempty_never_proposed_for_deletion(tmp_path):
    from oscardp.stage02.curation import investigate
    directory, _, _ = make_movie(tmp_path / 'analysis')
    original = tmp_path / 'source.srt'
    original.write_text(SRT + '\n3\n00:00:00,000 --> 00:00:00,000\nMeaningful dialogue.\n')
    report = validate_movie(directory, [original])
    result = investigate(report, tmp_path / 'review')
    assert result['proposals'][0]['type'] == 'RESTORE_TIMESTAMPS_FROM_AUTHORITATIVE_SOURCE'
    assert result['proposals'][0]['after'] is None
    assert not (tmp_path / 'review/proposed_sources').exists()


@pytest.mark.parametrize('newline', ['\n', '\r\n'])
def test_webvtt_diagnostic_does_not_invent_indices(newline):
    from oscardp.stage02.curation import webvtt_investigation
    text = 'WEBVTT\n\n00:00:00.500 --> 00:00:01.500\nHello.\n\n00:00:00.500 --> 00:00:01.500\nHello.'
    text = text.replace('\n', newline)
    refs = {88: [dict(shot_id='shot_000001', field='dialogue', indices=[88], text='Hello.')]}
    result = webvtt_investigation(text, refs, dict(shots=[dict(shot_id='shot_000001', temporal=dict(start=0, end=1))]))
    assert result['timed_block_count'] == 2 and result['unique_signature_count'] == 1
    assert not result['original_indices_recoverable']
    assert result['candidate_links'][0]['candidate_count'] == 1
    assert not result['candidate_links'][0]['proves_original_index']


def test_consolidated_file_patch_preserves_displaced_dialogue(tmp_path):
    from oscardp.stage02.curation import dump
    from oscardp.stage02.review import file_patch_bundles, patches_for
    directory, _, two = make_movie(tmp_path / 'analysis')
    text = '-♪ Song ♪ -Hello.'
    for shot in two['shots'][:2]:
        shot['song_lyrics'] = [dict(srt_index=1, text=text)]
    path = directory / f'Stage02_{ID}.json'
    path.write_text(json.dumps(two))
    srt = tmp_path / 'source.srt'
    srt.write_text('1\n00:00:00,500 --> 00:00:01,500\n' + text)
    report = validate_movie(directory, [srt])
    root = tmp_path / 'validation'
    dump(root / 'movies' / f'{ID}.json', report)
    before = path.read_bytes()
    bundles = file_patch_bundles(patches_for(report), root)
    bundle = bundles[0]
    assert len(bundle['companion_entries']) == 2
    assert all(e['value'] == dict(srt_indices=[1], speaker=None, text='Hello.') for e in bundle['companion_entries'])
    # Apply guarded operations only to an in-memory copy to verify the proposal.
    candidate = deepcopy(two)
    for op in bundle['json_patch']:
        parts = op['path'].strip('/').split('/')
        cursor = candidate
        for part in parts[:-1]:
            cursor = cursor[int(part)] if isinstance(cursor, list) else cursor[part]
        key = int(parts[-1]) if isinstance(cursor, list) else parts[-1]
        if op['op'] == 'test':
            assert cursor[key] == op['value']
        else:
            cursor[key] = deepcopy(op['value'])
    for old, new in zip(two['shots'], candidate['shots']):
        assert old['shot_id'] == new['shot_id'] and old['temporal'] == new['temporal']
    assert candidate['shots'][0]['song_lyrics'][0]['text'] == '♪ Song ♪'
    assert candidate['shots'][0]['dialogue'][0]['text'] == 'Hello.'
    assert path.read_bytes() == before


def test_complete_source_failure_has_one_error_with_full_diagnostics(tmp_path):
    source = '\n\n'.join(f'not an index {i}\nwrong timing\ntext' for i in range(100))
    report = fixture_movie(tmp_path, source, dict(dialogue=[]))
    assert report['summary']['errors'] == 1
    assert len(report['source_diagnostics']['parse_issues']) == 100
    assert report['issues'][0]['rule_code'] == 'SRT_PARSE_FAILED'


def test_review_rejects_changed_csv_evidence_even_with_same_counts(tmp_path):
    from oscardp.stage02.reporting import write_reports
    from oscardp.stage02.review import run_review
    directory, _, two = make_movie(tmp_path / 'analysis')
    (tmp_path / 'srts').mkdir()
    srt = tmp_path / 'srts/source.srt'
    srt.write_text(SRT)
    report = validate_movie(directory, [srt])  # Optional-cue INFO also has evidence.
    baseline, validation = tmp_path / 'baseline', tmp_path / 'validation'
    write_reports(baseline, [report], [], tolerance=1e-6)
    write_reports(validation, [report], [], tolerance=1e-6)
    csv_path = baseline / 'issues.csv'
    csv_path.write_text(csv_path.read_text().replace('Optional performance cues absent/empty', 'Changed evidence message'))
    with pytest.raises(ValueError, match='disagree'):
        run_review(baseline, validation, tmp_path / 'review')
    assert not (tmp_path / 'review').exists()
