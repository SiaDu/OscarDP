"""Real local speech activity and pretrained speech emotion inference."""
from pathlib import Path

import numpy as np

from .media import load_audio


def audio_affect(inp, output, model_name, model_dir):
    import torch
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

    torch.set_num_threads(4)
    model_dir = Path(model_dir)
    vad_path = model_dir / "silero_vad.jit"
    if not vad_path.exists():
        raise RuntimeError(f"Silero VAD checkpoint missing: {vad_path}; see Stage06 documentation")
    vad = torch.jit.load(str(vad_path), map_location="cpu").eval()
    vad.reset_states()
    waveform = load_audio(output / "scene_audio.wav")
    probabilities = []
    with torch.inference_mode():
        for i in range(0, len(waveform), 512):
            chunk = np.pad(waveform[i:i+512], (0, max(0, 512-len(waveform[i:i+512]))))
            probabilities.append(float(vad(torch.from_numpy(chunk), 16000).item()))
    # Conservative 32ms speech support; not forced alignment or speaker diarization.
    bins = []
    for i, prob in enumerate(probabilities):
        if prob >= .5:
            a, b = inp["start"]+i*.032, min(inp["end"], inp["start"]+(i+1)*.032)
            if bins and abs(bins[-1][1]-a) < 1e-6:
                bins[-1][1] = b
            else:
                bins.append([a,b])
    bins = [b for b in bins if b[1]-b[0] >= .128]
    # Checkpoints are prepared separately; inference never needs a network lookup.
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, cache_dir=str(model_dir / "huggingface"), local_files_only=True)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name, cache_dir=str(model_dir / "huggingface"), local_files_only=True).eval()
    proposals, skipped = [], []
    for cue in inp["subtitles"]:
        speech = [[max(a,cue["start"]), min(b,cue["end"])] for a,b in bins
                  if min(b,cue["end"]) > max(a,cue["start"])]
        if sum(b-a for a,b in speech) < .25:
            skipped.append({"srt_index": cue["srt_index"], "reason": "Less than 250ms VAD speech; no affect prediction"})
            continue
        start, end = speech[0][0], speech[-1][1]
        a, b = round((start-inp["start"])*16000), round((end-inp["start"])*16000)
        samples = waveform[a:b]
        if not len(samples) or float(np.max(np.abs(samples))) < 1e-4:
            skipped.append({"srt_index": cue["srt_index"], "reason": "Silent waveform; no affect prediction"})
            continue
        values = extractor(samples, sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            scores = model(**values).logits.softmax(-1)[0].cpu().tolist()
        label_scores = {model.config.id2label[i]: v for i,v in enumerate(scores)}
        label = max(label_scores, key=label_scores.get)
        affect = {"neu": "Neutral", "hap": "Happiness", "ang": "Anger", "sad": "Sadness"}.get(label, label)
        proposals.append(dict(proposal_id=f"audio_{cue['srt_index']:06d}", character_candidate=cue["speaker_candidate"],
            speaker_candidate=cue["speaker_candidate"], speaker_status=cue["speaker_status"], start=start, end=end,
            affect=affect, expression_style=None, source_modality="audio", score=label_scores[label],
            label_scores=label_scores, model_label=label, speech_intervals=speech,
            acoustic_evidence={"rms_dbfs": float(20*np.log10(max(float(np.sqrt(np.mean(samples**2))), 1e-10))),
                               "description": "Waveform RMS is descriptive loudness, not emotion evidence by itself."},
            evidence=[f"srt_{cue['srt_index']:06d}", "scene_audio.wav", "silero_vad_speech_support"],
            uncertainty=["Four-class English utterance classifier; film-domain accuracy unvalidated; softmax is not calibrated confidence.",
                         "Subtitle windows plus VAD are approximate; no forced alignment, separation or verified diarization.",
                         "Speaker candidate is text attribution; mixed/unknown speakers remain Unknown. Offscreen speech is not mapped to visible people."]))
    return dict(status="succeeded", model=model_name, inference="CPU waveform-only classification; subtitle times/speaker hints, no subtitle text sent to SER",
                vad_model="silero_vad.jit", vad_threshold=.5, vad_speech_intervals=bins,
                proposals=proposals, skipped=skipped, supported_labels=model.config.id2label,
                unsupported_dimensions=["valence", "arousal", "verified_speaker_identity"])
