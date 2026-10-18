"""Source-bound adjudication and narrowly evidenced presentation boundaries."""
from copy import deepcopy
import hashlib
import json

import pytest

from oscardp.stage02.adjudication import bind_decisions, check_consistency
from oscardp.stage02.srt import Cue
from oscardp.stage02.text import compare_entry
from oscardp.stage02.validator import validate_movie
from test_stage02_validator import make_movie, ID


@pytest.mark.parametrize('source,text', [
    ('-Oh. -♪ Song ♪', 'Oh.'),
    ('♪ Song ♪\n-MAN 2: This is awful!', 'This is awful!'),
    ('♪ Song ♪\n-WOMAN: My mom sings better!', 'My mom sings better!'),
    ('-I- -♪ Song ♪', 'I-'),
    ('-A-wim-a-weh. -♪ Song ♪', 'A-wim-a-weh.'),
    ('-2 degrees. -♪ Song ♪', '-2 degrees.'),
    ('♪ Song ♪ -Man, stop!', 'Man, stop!'),
])
def test_complete_boundaries(source, text):
    cue = Cue(1, 0, 2, source)
    assert compare_entry(text, [cue], 'dialogue')['status'] == 'exact'
    assert compare_entry(text[:-1], [cue], 'dialogue')['status'] == 'mismatch'


def record(cue):
    return dict(decision_id='D1', movie_id=ID, srt_index=1,
                original_srt_sha256='a'*64, source_text=cue.text, text='Hmm.',
                classification='sound_cues', affected_shots=['shot_000001'],
                source='user_confirmed_audiovisual_review', evidence='User viewed the film.')


def test_exact_human_binding_and_no_global_word_rule():
    cue = Cue(1, 0, 2, '♪ Song ♪ -Hmm.')
    decisions, errors = bind_decisions([record(cue)], ID, 'a'*64, [cue], [{'shot_id':'shot_000001'}])
    assert not errors
    assert compare_entry('Hmm.', [cue], 'sound_cues', decisions=decisions)['status'] == 'exact'
    assert compare_entry('Hmm.', [cue], 'sound_cues')['status'] == 'mismatch'
    assert compare_entry('Hmm.', [Cue(2, 0, 2, cue.text)], 'sound_cues', decisions=decisions)['status'] == 'mismatch'
    assert compare_entry('Hmm!', [cue], 'sound_cues', decisions=decisions)['status'] == 'mismatch'
    wrong = compare_entry('Hmm.', [cue], 'song_lyrics', decisions=decisions)
    assert wrong['classification_error']


@pytest.mark.parametrize('key,value', [
    ('original_srt_sha256','b'*64), ('source_text','Changed'), ('srt_index',2),
    ('classification','cheering'), ('source','codex_video_review'),
    ('text','Hmm'), ('affected_shots',['shot_999999']), ('evidence',''),
])
def test_invalid_binding_fails_closed(key,value):
    cue = Cue(1,0,2,'♪ Song ♪ -Hmm.')
    decision = record(cue); decision[key]=value
    bound, errors=bind_decisions([decision],ID,'a'*64,[cue],[{'shot_id':'shot_000001'}])
    assert not bound and errors


def test_cross_shot_repetition_and_duplicate_consistency():
    cue=Cue(1,0,2,'♪ Song ♪ -Hmm.'); decision=record(cue)
    decision['affected_shots'].append('shot_000002')
    matches=[dict(shot_id=s,field='sound_cues',entry=0,srt_indices=[1],
                  **compare_entry('Hmm.',[cue],'sound_cues',decisions=[decision]))
             for s in decision['affected_shots']]
    assert not check_consistency([decision],[],matches)
    assert check_consistency([decision],[],matches[:1])
    assert check_consistency([decision],[],matches+[deepcopy(matches[0])])


def test_adjudication_does_not_waive_other_checks(tmp_path):
    directory,one,two=make_movie(tmp_path/'analysis')
    source='1\n00:01:00,000 --> 00:01:02,000\n♪ Song ♪ -Hmm.\n'
    path=tmp_path/'source.srt';path.write_text(source)
    cue=Cue(1,60,62,'♪ Song ♪ -Hmm.');decision=record(cue)
    decision['original_srt_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    shot=two['shots'][0];shot['dialogue']=[]
    shot['sound_cues']=[dict(srt_index=1,text='Hmm.')]
    shot['temporal']['start']=999
    shot['unexpected']=True
    (directory/f'Stage02_{ID}.json').write_text(json.dumps(two))
    report=validate_movie(directory,[path],manual_decisions=[decision])
    codes={i['rule_code'] for i in report['issues']}
    assert 'SCHEMA_INVALID' in codes and 'TIME_INVALID' in codes and 'SRT_NO_OVERLAP' in codes
    assert report['summary']['validation_status']=='FAIL'


@pytest.mark.parametrize('source', ['-♪ Song ♪\n-Yeah!', '-Yeah!\n-♪ Song ♪'])
def test_singing_response_requires_specific_decision_and_keeps_lyrics(source):
    cue = Cue(1,0,2,source)
    decision = record(cue)
    decision.update(text='Yeah!', classification='song_lyrics')
    assert compare_entry('Yeah!',[cue],'song_lyrics')['status']=='mismatch'
    response=compare_entry('Yeah!',[cue],'song_lyrics',decisions=[decision])
    lyric=compare_entry('♪ Song ♪',[cue],'song_lyrics',decisions=[decision])
    assert response['status']=='exact' and response['human_decision_ids']==['D1']
    assert lyric['status']=='exact' and lyric['human_decision_ids']==[]
    assert compare_entry('Yeah!',[cue],'sound_cues',decisions=[decision])['classification_error']


def test_confirmed_utterance_does_not_infer_speaker():
    cue=Cue(1,0,2,'♪ Song ♪ -Whoo!')
    decision=record(cue);decision.update(text='Whoo!',classification='dialogue',speaker=None)
    match=dict(shot_id='shot_000001',field='dialogue',entry=0,srt_indices=[1],
               **compare_entry('Whoo!',[cue],'dialogue',decisions=[decision]))
    shots=[dict(shot_id='shot_000001',dialogue=[dict(text='Whoo!',speaker='Guessed man')])]
    assert check_consistency([decision],shots,[match])
    shots[0]['dialogue'][0]['speaker']=None
    assert not check_consistency([decision],shots,[match])
