"""Decode one continuous excerpt; all annotation times stay movie-global."""
import json
import subprocess
import wave
import textwrap

import numpy as np


def probe(path):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
                            check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def extract(inp, output):
    source = inp["sources"]["video"]
    info = probe(source)
    types = {s["codec_type"] for s in info["streams"]}
    if not {"audio", "video"} <= types:
        raise ValueError("Source must have real video and audio")
    if float(info["format"]["duration"]) < inp["end"]:
        raise ValueError("Selected range exceeds source duration")
    # Accurate input seeking with transcoding discards the keyframe preroll.
    common = ["ffmpeg", "-nostdin", "-v", "error", "-n", "-ss", str(inp["start"]), "-i", source,
              "-t", str(inp["duration"])]
    commands = [
        common + ["-map", "0:v:0", "-map", "0:a:0", "-vf", "scale=960:-2", "-c:v", "libx264",
                  "-preset", "fast", "-crf", "21", "-fps_mode", "passthrough", "-c:a", "aac", "-b:a", "160k",
                  "-movflags", "+faststart", str(output / "review_video.mp4")],
        common + ["-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(output / "scene_audio.wav")],
        ["ffmpeg", "-nostdin", "-v", "error", "-n", "-i", str(output / "review_video.mp4"), "-map", "0:v:0", "-an",
         "-c:v", "copy", str(output / "visual_only.mp4")]]
    for cmd in commands:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    review, silent = probe(output / "review_video.mp4"), probe(output / "visual_only.mp4")
    if any(s["codec_type"] == "audio" for s in silent["streams"]):
        raise ValueError("Visual branch received an audio stream")
    if abs(float(review["format"]["duration"]) - inp["duration"]) > .15:
        raise ValueError("Review clip duration differs from requested source range")
    return {"source_streams": [{k: s.get(k) for k in ("index", "codec_type", "codec_name", "time_base", "start_time", "avg_frame_rate")}
                               for s in info["streams"] if s["codec_type"] in ("audio", "video")],
            "review_duration": float(review["format"]["duration"]), "commands": commands,
            "review_player_mapping": "movie_time = player_time + selected start; frame precision limited by source PTS/encoding",
            "audio": "scene_audio.wav: mono 16kHz PCM", "video": "review_video.mp4: original audio; visual_only.mp4: no audio"}


def load_audio(path):
    with wave.open(str(path)) as stream:
        if stream.getframerate() != 16000 or stream.getnchannels() != 1 or stream.getsampwidth() != 2:
            raise ValueError("Expected 16kHz mono PCM16 audio")
        return np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(np.float32) / 32768.


def frame_sheets(output, duration, fps=2.):
    """Cover the continuous clip, masking burned-in caption text before inspection."""
    filters=(f"fps={fps},drawbox=x=0:y=ih*0.78:w=iw:h=ih*0.22:color=black:t=fill,"
             "scale=400:-2,drawtext=text=%{pts\\:flt}:x=8:y=8:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.75,tile=4x5")
    subprocess.run(["ffmpeg","-nostdin","-v","error","-n","-i",str(output/"visual_only.mp4"),
                    "-vf",filters,"-fps_mode","vfr",str(output/"observed_frames_%02d.jpg")],check=True,capture_output=True)
    subprocess.run(["ffmpeg","-nostdin","-v","error","-n","-ss",str(max(0.,duration-.098417)),
                    "-i",str(output/"visual_only.mp4"),"-vf","drawbox=x=0:y=ih*0.78:w=iw:h=ih*0.22:color=black:t=fill,scale=800:-2",
                    "-frames:v","1",str(output/"observed_tail.jpg")],check=True,capture_output=True)


def annotated_review(output):
    """Place proposals in a side panel so the actor's face remains unobscured."""
    data=json.loads((output/"stage06_prototype.json").read_text())
    video=next(s for s in probe(output/"review_video.mp4")["streams"] if s["codec_type"]=="video")
    w,h=video["width"],video["height"]
    panel=480
    def stamp(s):
        centis=round(s*100)
        hours,rem=divmod(centis,360000)
        minutes,rem=divmod(rem,6000)
        sec,cs=divmod(rem,100)
        return f"{hours}:{minutes:02d}:{sec:02d}.{cs:02d}"
    ass=("[Script Info]\nScriptType: v4.00+\n"+f"PlayResX: {w+panel}\nPlayResY: {h}\nWrapStyle: 2\n\n"
         "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
         "Style: Default,DejaVu Sans,17,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1\n\n"
         "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
    start,end=data["time_range"]["start"],data["time_range"]["end"]
    cursor=start
    while cursor < end:
        stop=min(cursor+1,end)
        lines=[f"MOVIE {cursor:.3f} - {stop:.3f}s","MODEL PROPOSALS / NOT GROUND TRUTH",""]
        for state in data["affect_states"]:
            if state["start"] >= stop or state["end"] <= cursor:
                continue
            cov=state["observation_coverage"]
            observed=[m for m in ("audio","video") if any(a<stop and b>cursor for a,b in cov[m])]
            lines += [f"{state['character_candidate']}: {state['affect']}? / {state['expression_style']}",state["state_id"],
                      "Evidence now: "+(" + ".join(observed) if observed else "NO AV; inferred continuity"),""]
            tr=state["transition"]
            if tr and tr["start"] < stop and tr["end"] > cursor:
                lines += ["Uncertain transition interval",""]
        local=[e for e in data["events"] if e["start"]<stop and e["end"]>cursor]
        lines += ["UNREVIEWED EVENT CANDIDATES"]
        lines += [f"{e['event_id']}  {e['feature']}" for e in local[:5]] if local else ["No measured event in this cue"]
        if len(local)>5:
            lines += [f"+ {len(local)-5} more; see JSON"]
        wrapped=[part for line in lines for part in (textwrap.wrap(line,44) or [""])]
        text="\\N".join(wrapped)
        ass+=f"Dialogue: 0,{stamp(cursor-start)},{stamp(stop-start)},Default,,0,0,0,,{{\\an7\\pos({w+18},18)}}{text}\n"
        cursor=stop
    subtitle=output/"review_overlay.ass"
    with subtitle.open("x",encoding="utf-8") as stream:
        stream.write(ass)
    escaped=str(subtitle.resolve()).replace("\\", "\\\\").replace(":", "\\:").replace("'", "'\\''")
    subprocess.run(["ffmpeg","-nostdin","-v","error","-n","-i",str(output/"review_video.mp4"),
                    "-map","0:v:0","-map","0:a:0","-vf",f"pad=iw+{panel}:ih:0:0:color=black,ass=filename='{escaped}'",
                    "-c:v","libx264","-preset","fast","-crf","21","-fps_mode","passthrough","-c:a","copy",
                    "-map_metadata","-1","-map_chapters","-1","-movflags","+faststart",str(output/"review_video_annotated.mp4")],
                   check=True,capture_output=True)
