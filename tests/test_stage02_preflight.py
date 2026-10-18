"""Small source fixtures reproduce failure mechanisms without copying film subtitles."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from oscardp.stage02.__main__ import main
from oscardp.stage02.preflight import run_preflight
from test_stage02_validator import make_movie

ID='tt7654321'
GOOD='7\n00:00:00,500 --> 00:00:01,500\nA test caption.\n'


def pdf(path, *, text=True, encrypted=False, extra_blank_pages=0):
    writer=PdfWriter()
    page=writer.add_blank_page(612,792)
    if text:
        font=DictionaryObject({NameObject('/Type'):NameObject('/Font'),NameObject('/Subtype'):NameObject('/Type1'),NameObject('/BaseFont'):NameObject('/Helvetica')})
        page[NameObject('/Resources')]=DictionaryObject({NameObject('/Font'):DictionaryObject({NameObject('/F1'):writer._add_object(font)})})
        stream=DecodedStreamObject()
        stream.set_data(b'BT /F1 12 Tf 30 700 Td ('+b'A synthetic screenplay scene with useful continuous dialogue context. '*8+b') Tj ET')
        page[NameObject('/Contents')]=writer._add_object(stream)
    for _ in range(extra_blank_pages):
        writer.add_blank_page(612,792)
    if encrypted:
        writer.encrypt('test-only-password')
    with path.open('wb') as f:
        writer.write(f)


@pytest.fixture
def source(tmp_path):
    directory,one,two=make_movie(tmp_path/'inputs',movie_id=ID)
    srt=directory/f'{ID}.srt'
    srt.write_text(GOOD)
    screenplay=directory/f'{ID}.pdf'
    pdf(screenplay)
    return dict(stage01=directory/f'Stage01_{ID}.json',srt=srt,screenplay=screenplay,
                stage02_path=directory/f'Stage02_{ID}.json',one=one,two=two,root=tmp_path/'reports')


def run(source,name='first',**kwargs):
    return run_preflight(**{k:source[k] for k in ['stage01','srt','screenplay']},report_root=source['root']/name,**kwargs)


def codes(report):
    return {i['rule_code'] for i in report['issues']}


def duration(source,end):
    one=deepcopy(source['one'])
    one['shots']=[dict(shot_id='shot_000001',temporal=dict(start=0,end=end,start_frame=0,end_frame=int(end*24)))]
    source['stage01'].write_text(json.dumps(one))


def approval(report,path,issue=None,**changes):
    issue=issue or next(i for i in report['issues'] if i['requires_approval'])
    record=dict(issue_id=issue['issue_id'],rule_code=issue['rule_code'],movie_id=report['movie_id'],
        input_sha256={k:v['sha256'] for k,v in report['inputs'].items()},approved_by='Fixture reviewer',
        evidence='Manually reviewed this precise harmless source exception.',decision='APPROVED_NONBLOCKING')
    record.update(changes)
    path.write_text(json.dumps(dict(schema_version='1.0',exceptions=[record])))
    return path


def test_good_sources_real_cli_and_provenance(source):
    args=['preflight']
    for k in ['stage01','srt','screenplay']:
        args += ['--'+k,str(source[k])]
    args+=['--report-root',str(source['root']/'cli')]
    assert main(args)==0
    report=json.loads((source['root']/'cli/source_preflight.json').read_text())
    assert report['preflight_status']=='PASS' and report['llm_annotation_allowed']
    assert all(report['check_completeness'].values())
    for k in ['Stage01','SRT','Screenplay']:
        info=report['inputs'][k]
        assert info['sha256']==hashlib.sha256(Path(info['path']).read_bytes()).hexdigest()
        assert info['bytes']>0 and info['mtime_ns']>0
    assert report['screenplay_readability']['page_count']==1
    assert report['screenplay_readability']['screenplay_movie_match']=='UNVERIFIED'
    assert report['srt_format_and_integrity']['cue_count']==1
    assert (source['root']/'cli/PREFLIGHT.md').is_file()
    assert (source['root']/'cli/issues.csv').is_file()


@pytest.mark.parametrize('source_text,expected',[
    (GOOD+'\n7\n00:00:02,000 --> 00:00:02,500\nSecond content.', 'SRT_DUPLICATE_INDEX'),
    ('WEBVTT\n\n00:00:00.500 --> 00:00:01.500\nA caption.','SRT_UNSUPPORTED_FORMAT'),
    ('00:00:00.500 --> 00:00:01.500\nA caption.','SRT_UNSUPPORTED_FORMAT'),
    ('7\n00:00:00,000 --> 00:00:00,000\nMeaningful text.','SRT_ZERO_DURATION'),
    ('7\n00:00:02,000 --> 00:00:01,000\nMeaningful text.','SRT_REVERSED_TIME'),
    ('0\n00:00:01,000 --> 00:00:02,000\nText.','SRT_TIMESTAMP'),
    ('No structured subtitles here.','SRT_UNSUPPORTED_FORMAT'),
    ('2\nnot a timestamp\nText.','SRT_TIMESTAMP'),
    ('00:00:01,000 --> 00:00:02,000\nText.','SRT_FORMAT'),
])
def test_subtitle_integrity_blockers(source,source_text,expected):
    source['srt'].write_text(source_text)
    before=source['srt'].read_bytes()
    report=run(source)
    assert report['preflight_status']=='BLOCKED' and not report['llm_annotation_allowed']
    assert expected in codes(report)
    assert source['srt'].read_bytes()==before


@pytest.mark.parametrize('raw,expected',[
    (b'\xffbroken','SRT_UNDECODABLE'),
    ('text'.encode('utf-16'),'SRT_ENCODING_UNSUPPORTED'),
    (GOOD.replace('test','\ufffd').encode(),'SRT_ENCODING_CORRUPT'),
])
def test_encoding_errors_are_not_guessed(source,raw,expected):
    source['srt'].write_bytes(raw)
    report=run(source)
    assert report['exit_code']==1 and expected in codes(report)


def test_bom_and_crlf_are_supported_without_rewriting(source):
    raw=b'\xef\xbb\xbf'+GOOD.replace('\n','\r\n').encode()
    source['srt'].write_bytes(raw)
    report=run(source)
    assert report['exit_code']==0
    assert report['srt_format_and_integrity']['bom']=='UTF-8'
    assert report['srt_format_and_integrity']['newline_counts']['CRLF']==3
    assert source['srt'].read_bytes()==raw


@pytest.mark.parametrize('middle',[(1839.12,1840.62),(9000.12,9001.62)])
def test_isolated_timestamp_outlier_in_other_movie(source,middle):
    duration(source,10000)
    def stamp(value):
        h,rem=divmod(value,3600);m,s=divmod(rem,60)
        return f'{int(h):02}:{int(m):02}:{s:06.3f}'.replace('.',',')
    text='\n\n'.join(f'{index}\n{stamp(start)} --> {stamp(end)}\nSynthetic line {index}.'
        for index,start,end in [(17,5433.41,5434.5),(18,*middle),(19,5441.41,5443.37)])
    source['srt'].write_text(text)
    report=run(source)
    assert report['preflight_status']=='BLOCKED'
    assert 'SRT_TIMELINE_ISOLATED_OUTLIER' in codes(report)
    issue=next(i for i in report['issues'] if i['rule_code']=='SRT_TIMELINE_ISOLATED_OUTLIER')
    assert issue['evidence']['previous']['index']==17 and issue['evidence']['following']['index']==19
    assert source['srt'].read_text()==text


def test_long_nested_and_normal_overlaps_allowed(source):
    duration(source,100)
    source['srt'].write_text('1\n00:00:35,000 --> 00:00:39,000\nFirst.\n\n2\n00:00:01,000 --> 00:00:40,000\nOverlapping context.\n\n3\n00:00:39,000 --> 00:00:40,000\nNext.')
    report=run(source)
    assert report['preflight_status']=='PASS'
    assert report['subtitle_timeline']['overlaps_allowed']


@pytest.mark.parametrize('text,code',[
    ('8\n00:00:01,500 --> 00:00:02,500\n','SRT_EMPTY_CUE'),
    ('8\n00:00:01,500 --> 00:00:02,500\nVisit www.example.org today.','SRT_SUSPECT_ADVERTISING'),
])
def test_nonblocking_risks_still_require_explicit_review(source,text,code):
    source['srt'].write_text(GOOD+'\n'+text)
    before=source['srt'].read_bytes()
    report=run(source)
    assert report['preflight_status']=='WARNING' and not report['llm_annotation_allowed']
    assert report['exit_code']==1 and code in codes(report)
    issue=next(i for i in report['issues'] if i['rule_code']==code)
    assert issue['requires_approval'] and 'raw' in issue['evidence']['block']
    assert source['srt'].read_bytes()==before


def test_reviewable_warning_approved_but_status_preserved(source,tmp_path):
    source['srt'].write_text(GOOD+'\n8\n00:00:02,000 --> 00:00:02,500\n')
    first=run(source)
    path=approval(first,tmp_path/'approved.json')
    second=run(source,'approved',approved_exceptions=path)
    assert second['preflight_status']=='WARNING'
    assert second['llm_annotation_allowed'] and second['exit_code']==0
    assert first['issues'][0]['issue_id']==second['issues'][0]['issue_id']


@pytest.mark.parametrize('change', ['hash','id','rule','movie','blanket','duplicate'])
def test_bad_approval_cannot_open_gate(source,tmp_path,change):
    source['srt'].write_text(GOOD+'\n8\n00:00:02,000 --> 00:00:02,500\n')
    report=run(source)
    path=approval(report,tmp_path/'approved.json')
    data=json.loads(path.read_text());record=data['exceptions'][0]
    if change=='hash':record['input_sha256']['SRT']='b'*64
    elif change=='id':record['issue_id']='some_other_issue'
    elif change=='rule':record['rule_code']='another_rule'
    elif change=='movie':record['movie_id']='tt8888888'
    elif change=='blanket':record['decision']='APPROVED_ALL'
    else:data['exceptions'].append(deepcopy(record))
    path.write_text(json.dumps(data))
    final=run(source,'rejected',approved_exceptions=path)
    assert not final['llm_annotation_allowed'] and 'APPROVAL_BINDING_INVALID' in codes(final)


def test_integrity_blocker_cannot_be_waived(source,tmp_path):
    source['srt'].write_text('1\n00:00:00,000 --> 00:00:00,000\nMeaningful.')
    first=run(source)
    issue=next(i for i in first['issues'] if i['rule_code']=='SRT_ZERO_DURATION')
    path=approval(first,tmp_path/'approval.json',issue=issue)
    final=run(source,'rejected',approved_exceptions=path)
    assert final['preflight_status']=='BLOCKED' and not final['llm_annotation_allowed']
    assert 'APPROVAL_BINDING_INVALID' in codes(final)


@pytest.mark.parametrize('variant',['missing','bad','scanned','encrypted','sparse'])
def test_pdf_failures_and_sparse_layer(source,variant):
    if variant=='missing':source['screenplay'].unlink()
    elif variant=='bad':source['screenplay'].write_bytes(b'%PDF-1.7\nThis is damaged.')
    elif variant=='scanned':pdf(source['screenplay'],text=False)
    elif variant=='encrypted':pdf(source['screenplay'],encrypted=True)
    elif variant=='sparse':pdf(source['screenplay'],extra_blank_pages=3)
    report=run(source)
    assert not report['llm_annotation_allowed']
    assert report['preflight_status']==('WARNING' if variant=='sparse' else 'BLOCKED')
    assert not report['screenplay_readability']['ocr_performed']
    assert report['screenplay_readability']['screenplay_movie_match']=='UNVERIFIED'


@pytest.mark.parametrize('mutation', ['gap','frames','order','duplicate','negative','non_object','empty','root_keys','identity','nan','duplicate_json_key'])
def test_stage01_contract_is_not_weakened(source,mutation):
    one=deepcopy(source['one'])
    if mutation=='gap':one['shots'][1]['temporal']['start']=1.1
    elif mutation=='frames':one['shots'][1]['temporal']['start_frame']=100
    elif mutation=='order':one['shots'].reverse()
    elif mutation=='duplicate':one['shots'][1]['shot_id']=one['shots'][0]['shot_id']
    elif mutation=='negative':one['shots'][0]['temporal']['start']=-1
    elif mutation=='non_object':one['shots'][0]=3
    elif mutation=='empty':one['shots']=[]
    elif mutation=='root_keys':one['extra']=True
    elif mutation=='identity':one['movie_id']='tt8888888'
    elif mutation=='nan':one['shots'][0]['temporal']['end']=float('nan')
    text=json.dumps(one)
    if mutation=='duplicate_json_key':text=text.replace('"movie_id":','"movie_id": "tt0000000", "movie_id":',1)
    source['stage01'].write_text(text)
    report=run(source)
    assert report['preflight_status']=='BLOCKED' and not report['llm_annotation_allowed']


def test_source_version_changed_is_explicit_and_old_manifest_immutable(source,tmp_path):
    baseline=tmp_path/'old_manifest.json'
    baseline.write_text(json.dumps(dict(movies=[dict(movie_id=ID,original_srt_path=str(source['srt']),original_srt_sha256='a'*64)])))
    old=baseline.read_bytes();current=source['srt'].read_bytes()
    report=run(source,baseline_manifests=[baseline])
    assert 'SOURCE_VERSION_CHANGED' in codes(report)
    assert report['inputs']['SRT']['sha256']==hashlib.sha256(current).hexdigest()
    assert baseline.read_bytes()==old and source['srt'].read_bytes()==current
    assert not report['llm_annotation_allowed']


def test_validator_movie_report_can_supply_explicit_baseline(source,tmp_path):
    baseline=tmp_path/'validator_movie.json'
    baseline.write_text(json.dumps(dict(summary=dict(movie_id=ID),inputs=dict(SRT=dict(path=str(source['srt']),sha256='a'*64)))))
    report=run(source,baseline_manifests=[baseline])
    assert 'SOURCE_VERSION_CHANGED' in codes(report)
    assert report['source_version_comparison']['comparisons'][0]['historical_sha256']=='a'*64


def test_existing_stale_reference_reports_actual_shot_and_candidates(source):
    two=deepcopy(source['two'])
    two['shots'][2]['dialogue']=[dict(srt_indices=[7],speaker=None,text='A test caption.')]
    source['stage02_path'].write_text(json.dumps(two))
    before=source['stage02_path'].read_bytes()
    report=run(source,stage02=source['stage02_path'])
    stale=[i for i in report['issues'] if i['rule_code']=='STAGE02_STALE_CUE_REFERENCE']
    assert len(stale)==1 and stale[0]['evidence']['shot_id']=='shot_000003'
    assert stale[0]['evidence']['overlapping_stage01_candidates']==['shot_000001','shot_000002']
    assert not report['llm_annotation_allowed']
    assert source['stage02_path'].read_bytes()==before


def test_existing_text_and_missing_indices_block(source):
    two=deepcopy(source['two'])
    two['shots'][0]['dialogue']=[dict(srt_indices=[7],speaker=None,text='Different text.'),dict(srt_indices=[99],speaker=None,text='Missing.')]
    source['stage02_path'].write_text(json.dumps(two))
    report=run(source,stage02=source['stage02_path'])
    assert {'STAGE02_TEXT_MISMATCH','STAGE02_SRT_INDEX_MISSING'}<=codes(report)


def test_empty_referenced_cue_is_blocking(source):
    source['srt'].write_text('7\n00:00:00,500 --> 00:00:01,500\n')
    two=deepcopy(source['two']);two['shots'][0]['dialogue']=[dict(srt_indices=[7],text='Untraceable.',speaker=None)]
    source['stage02_path'].write_text(json.dumps(two))
    report=run(source,stage02=source['stage02_path'])
    assert next(i for i in report['issues'] if i['rule_code']=='SRT_EMPTY_CUE')['severity']=='BLOCKED'


def test_overshoot_and_possible_whole_track_offset(source):
    source['srt'].write_text('1\n00:01:00,000 --> 00:01:01,000\nA.\n\n2\n00:01:02,000 --> 00:01:03,000\nB.')
    report=run(source)
    assert 'SRT_OUTSIDE_MOVIE' in codes(report) and report['preflight_status']=='BLOCKED'


def test_delayed_entire_track_is_a_review_risk_not_guessed_shift(source):
    duration(source,1000)
    source['srt'].write_text('1\n00:15:00,000 --> 00:15:01,000\nA.\n\n2\n00:15:02,000 --> 00:15:03,000\nB.\n\n3\n00:15:04,000 --> 00:15:05,000\nC.')
    report=run(source)
    assert 'SRT_POSSIBLE_OFFSET_OR_INCOMPLETE' in codes(report)
    assert report['preflight_status']=='WARNING' and not report['llm_annotation_allowed']
    assert report['subtitle_timeline']['global_offset_seconds'] is None


def test_existing_schema_errors_and_changed_stage01_cannot_be_approved(source,tmp_path):
    source['srt'].write_text(GOOD+'\n8\n00:00:02,000 --> 00:00:02,500\n')
    first=run(source)
    path=approval(first,tmp_path/'approved.json')
    source['stage01'].write_text(source['stage01'].read_text()+'\n')
    final=run(source,'changed_stage01',approved_exceptions=path)
    assert not final['llm_annotation_allowed'] and 'APPROVAL_BINDING_INVALID' in codes(final)
    two=deepcopy(source['two']);two['unknown_field']=True
    source['stage02_path'].write_text(json.dumps(two))
    report=run(source,'bad_schema',stage02=source['stage02_path'])
    assert 'STAGE02_SCHEMA_INVALID' in codes(report) and not report['llm_annotation_allowed']


def test_pdf_partial_failure_is_not_marked_complete(source,monkeypatch):
    from pypdf import PageObject
    def fail(*args,**kwargs):
        raise ValueError('Synthetic page extraction failure')
    monkeypatch.setattr(PageObject,'extract_text',fail)
    report=run(source)
    assert 'SCREENPLAY_EXTRACTION_FAILED' in codes(report)
    assert not report['check_completeness']['screenplay_readability']
    assert not report['llm_annotation_allowed']


def test_missing_cue_number_in_numbered_structure_is_blocked(source):
    source['srt'].write_text('1\n00:00:00,100 --> 00:00:00,300\nHello.\n\n00:00:01,000 --> 00:00:02,000\nMissing number.')
    report=run(source)
    assert 'SRT_FORMAT' in codes(report) and report['preflight_status']=='BLOCKED'


def test_five_historical_failure_mechanisms_without_dataset_dependency(tmp_path):
    cases=[('tt0025316',GOOD+'\n7\n00:00:02,000 --> 00:00:02,500\nAdvertise your product here.','SRT_DUPLICATE_INDEX'),
           ('tt14961016',GOOD+'\n7\n00:00:02,000 --> 00:00:02,500\nVisit www.example.org','SRT_DUPLICATE_INDEX'),
           ('tt26581740','1\n00:00:00,000 --> 00:00:00,100\n\n2\n00:00:00,200 --> 00:00:00,300\n\n'+GOOD,'SRT_EMPTY_CUE'),
           ('tt27714581','419\n00:00:00,000 --> 00:00:00,000\nAn independent historical zero-time fixture.','SRT_ZERO_DURATION'),
           ('tt18382850','WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nIndependent example.','SRT_UNSUPPORTED_FORMAT')]
    for mid,text,expected in cases:
        directory,_,_=make_movie(tmp_path/'inputs',movie_id=mid)
        path=directory/f'{mid}.srt';path.write_text(text)
        script=directory/f'{mid}.pdf';pdf(script)
        report=run_preflight(stage01=directory/f'Stage01_{mid}.json',srt=path,screenplay=script,report_root=tmp_path/'reports'/mid)
        assert expected in codes(report) and not report['llm_annotation_allowed']


def test_existing_reports_and_source_directories_are_protected(source):
    report=run(source)
    path=source['root']/'first/source_preflight.json';before=path.read_bytes()
    with pytest.raises(FileExistsError):run(source)
    assert path.read_bytes()==before
    with pytest.raises(ValueError):
        run_preflight(**{k:source[k] for k in ['stage01','srt','screenplay']},report_root=source['srt'].parent/'report')
    assert report['preflight_status']=='PASS'


def test_cli_config_error_exit_two(source,tmp_path):
    output=source['root']/'occupied';output.mkdir(parents=True);(output/'keep').write_text('preserved')
    assert main(['preflight','--stage01',str(source['stage01']),'--srt',str(source['srt']),
        '--screenplay',str(source['screenplay']),'--report-root',str(output)])==2
    assert (output/'keep').read_text()=='preserved'
