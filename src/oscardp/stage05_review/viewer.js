"use strict";
const $ = id => document.getElementById(id);
const pretty = x => JSON.stringify(x, null, 2);
let index, current, shot, frames, traces, timeline, overlay, revision, windowEnd = null;
let loadGeneration = 0;
const cache = new Map();
async function get(url) {
  const r = await fetch(url, {cache: "no-store"});
  if (!r.ok) throw Error((await r.text()) || r.statusText);
  return r.json();
}
async function data(url) {
  if (!cache.has(url)) cache.set(url, get(url));
  return cache.get(url);
}
function fail(e) { $("saveStatus").textContent = "未保存 / Error: " + e.message; }
function reportProgress() {
  const done = overlay.records.filter(r => r.review_status === "human_reviewed").length;
  $("progress").textContent = `${done} / ${index.items.length} 人工复核记录完成；候选文件保持原样`;
}
async function restore() {
  const saved = await get("/api/reviews");
  if (saved.source_run_sha256 !== index.source_run_sha256) throw Error("Source Hash mismatch");
  revision = saved.revision;
  overlay = {schema_version: saved.schema_version, source_run_sha256: saved.source_run_sha256, records: saved.records};
  $("status").textContent = "来源已匹配 · 审核进度文件：" + saved.review_file;
  reportProgress();
  if (current) fillForm();
}
function fillQueue() {
  const previous = current?.review_id;
  $("queue").replaceChildren();
  for (const i of index.items) {
    if ($("scope").value === "pilot" && !i.pilot_selected) continue;
    if ($("shotFilter").value !== "all" && $("shotFilter").value !== i.shot_id) continue;
    const q = i.source_queue_item;
    const option = new Option(`${i.review_id} · ${i.shot_id.slice(-3)} · ${q.feature || q.event_type} · ${q.start_frame}–${q.end_frame}`, i.review_id);
    $("queue").add(option);
  }
  if ([...$("queue").options].some(o => o.value === previous)) $("queue").value = previous;
  if (!$("queue").value && $("queue").options.length) $("queue").selectedIndex = 0;
  if ($("queue").value) choose().catch(fail);
}
async function choose() {
  const generation = ++loadGeneration;
  $("video").pause();
  current = index.items.find(i => i.review_id === $("queue").value);
  shot = index.packet.scenes.flatMap(s => s.shots).find(s => s.shot_id === current.shot_id);
  const sid = current.shot_id;
  const loaded = await Promise.all([data(`data/${sid}_frames.json`), data(`data/${sid}_traces.json`), data(`data/${sid}_timeline.json`)]);
  if (generation !== loadGeneration) return;
  [frames, traces, timeline] = loaded;
  windowEnd = null;
  $("video").src = `media/${sid}.mp4`;
  $("still").hidden = true;
  $("video").hidden = false;
  const q = current.source_queue_item;
  $("title").textContent = `${current.review_id} · ${sid} · ${q.sample_kind}`;
  const event = index.packet.event_candidates.find(e => e.event_id === current.source_event_id);
  $("detail").textContent = pretty({queue: q, event: event || null, pilot_selected: current.pilot_selected,
    source_run_sha256: index.source_run_sha256});
  const scene = index.packet.scenes.find(s => s.scene_id === current.scene_id);
  $("context").textContent = pretty({scene: scene.context, dialogue: shot.stage02_dialogue,
    narrative: shot.stage02_narrative, stage04_candidates: shot.stage04_character_candidates,
    face_tracks: shot.face_tracks, effective_processing_temporal: shot.effective_processing_temporal,
    old_stage04_time: shot.stage04_context_temporal, boundary_review: shot.boundary_review,
    warning: "Dialogue Speaker ≠ visual identity; candidate character mapping unverified"});
  $("feature").replaceChildren();
  for (const feature of [...new Set(traces.map(t => t.feature))].sort()) $("feature").add(new Option(feature, feature));
  const inferred = q.feature === "AU45" ? "AU45_r" : q.feature || ({eye_closure_candidate: "eye_geometry",
    head_orientation_change: "head_orientation", camera_relative_gaze_change: "gaze_direction"}[q.event_type]);
  if ([...$("feature").options].some(o => o.value === inferred)) $("feature").value = inferred;
  $("frame").min = frames.frames[0].global_frame_index;
  $("frame").max = frames.frames.at(-1).global_frame_index;
  fillForm(); drawTimeline(); drawCurve();
  seekFrame(Number(q.start_frame), true);
}
function frameAtPreview(time) {
  return frames.frames[Math.min(frames.frames.length-1, Math.max(0, Math.floor(time*frames.preview_fps+1e-5)))];
}
function labelFrame(f) {
  $("frame").value = f.global_frame_index;
  $("frameLabel").textContent = `原始帧 ${f.global_frame_index} · movie ${f.movie_timestamp_sec.toFixed(6)} s · preview ${f.preview_time_sec.toFixed(6)} s · source ${frames.source_timestamp_rule}`;
}
function seekFrame(n, still = true) {
  if (!frames) return;
  const f = frames.frames.find(f => f.global_frame_index === Math.max(+$("frame").min, Math.min(+$("frame").max, n)));
  if (!f) return;
  $("video").pause();
  // Seek inside the encoded frame; exact still stepping never depends on browser seeking.
  $("video").currentTime = f.preview_time_sec + 0.1/frames.preview_fps;
  $("still").src = f.still;
  $("still").hidden = !still;
  $("video").hidden = still;
  labelFrame(f);
}
function drawTimeline() {
  const c = $("timeline"), ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  const start = frames.frames[0].global_frame_index, end = frames.frames.at(-1).global_frame_index+1;
  const kinds = ["au_activation", "eye_closure_candidate", "head_orientation_change", "camera_relative_gaze_change"];
  kinds.forEach((type, i) => { ctx.fillStyle = "#b6c9d6"; ctx.fillText(type, 8, 20+i*40); });
  for (const r of timeline.review_intervals) {
    const lane = kinds.indexOf(r.event_type);
    if (lane < 0) continue;
    const x = 280+(+r.start_frame-start)/(end-start)*700;
    const w = Math.max(2, (+r.end_frame-r.start_frame)/(end-start)*700);
    ctx.fillStyle = r.review_id === current.review_id ? "#ffbd61" : r.event_id ? "#57b6ac88" : "#8d98bb55";
    ctx.fillRect(x, lane*40+8, w, 17);
  }
  ctx.fillStyle = "#b6c9d6"; ctx.fillText(`${start} inclusive → ${end} exclusive`, 280, 176);
}
function drawCurve() {
  const selected = traces.filter(t => t.track_id === current.track_id && t.feature === $("feature").value);
  const c = $("curve"), ctx = c.getContext("2d"); ctx.clearRect(0, 0, c.width, c.height);
  const start = frames.frames[0].global_frame_index, end = frames.frames.at(-1).global_frame_index;
  const x = f => 55+(f-start)/Math.max(1,end-start)*920;
  const colors = ["#62d5c8", "#f0b468", "#b49be4", "#fa8b9b", "#a6d965", "#71b8ee"];
  ["raw_values", "processed_values"].forEach((field, panel) => {
    const top = panel*165+18, bottom = top+125;
    const values = selected.flatMap(r => r[field] || []).filter(Number.isFinite);
    let lo = Math.min(...values), hi = Math.max(...values);
    if (!values.length) {lo=0; hi=1;}
    if (hi === lo) hi=lo+1;
    ctx.fillStyle = "#b6c9d6"; ctx.fillText(`${field}  ${lo.toPrecision(3)} … ${hi.toPrecision(3)}`, 10, top);
    const a = +current.source_queue_item.start_frame, b = +current.source_queue_item.end_frame;
    ctx.fillStyle = "#edb86018"; ctx.fillRect(x(a), top+6, x(b)-x(a), 120);
    let color = 0;
    for (const mode of [...new Set(selected.map(r => r.extraction_mode))]) {
      const rows = selected.filter(r => r.extraction_mode === mode);
      const dimensions = Math.max(0, ...rows.map(r => (r[field] || []).length));
      for (let d=0; d<dimensions; d++) {
        ctx.strokeStyle = colors[color++ % colors.length];
        ctx.setLineDash(mode === "dynamic" ? [4,3] : []);
        ctx.beginPath(); let connected = false, previous = null;
        for (const r of rows) {
          const v = r[field]?.[d];
          if (!r.valid || !Number.isFinite(v)) {connected=false; previous=null; continue;}
          const px=x(r.global_frame_index), py=bottom-(v-lo)/(hi-lo)*110;
          if (connected && r.global_frame_index === previous+1) ctx.lineTo(px, py); else ctx.moveTo(px, py);
          connected=true; previous=r.global_frame_index;
        }
        ctx.stroke(); ctx.fillStyle=ctx.strokeStyle;
        ctx.fillText(`${mode}[${d}]`, 510+(color-1)%6*76, top);
      }
    }
    ctx.setLineDash([]);
    for (const r of selected.filter(r => !r.valid)) {
      ctx.fillStyle="#fb677a"; ctx.fillRect(x(r.global_frame_index), bottom, 3, 7);
    }
  });
  const f = $("feature").value;
  $("curveLabel").textContent = (f === "eye_geometry" ? "Raw=EAR；processed=relative EAR，两栏量纲不同。" : f.startsWith("AU") ? "Raw/processed=OpenFace AU intensity，非情绪。" : "Raw/processed=模型角度 radians；derived speed=rad/s。") + "静态实线 / 动态虚线；红标=invalid；缺失处不连线。";
  $("values").textContent = selected.map(r => `${r.global_frame_index}  ${r.movie_timestamp_sec.toFixed(6)}s  ${r.extraction_mode}  valid=${r.valid}  raw=${JSON.stringify(r.raw_values)}  processed=${JSON.stringify(r.processed_values)}  speed=${r.derived_value}`).join("\n");
}
function fillForm() {
  const r = overlay.records.find(r => r.review_id === current.review_id) || current.human_review;
  $("reviewer").value = r.reviewer || ""; $("observed").value=r.observed_event_label || "";
  $("presence").value=r.event_presence_decision || ""; $("boundary").value=r.temporal_boundary_decision || "";
  $("quality").value=r.evidence_quality || ""; $("notes").value=r.notes || "";
  $("reviewStart").value=r.reviewed_frame_range[0]; $("reviewEnd").value=r.reviewed_frame_range[1];
  for (const phase of ["onset", "apex", "offset"]) $(phase).value=r.suggested_boundaries?.[phase+"_frame"] ?? "";
  $("attest").checked=false;
  $("saveStatus").textContent = r.review_status + (r.reviewed_at ? " · "+r.reviewed_at : "");
}
async function save(records) {
  const response = await fetch("/api/reviews", {method:"POST", headers:{"Content-Type":"application/json", "X-OscarDP-Review":"1"},
    body:JSON.stringify({...overlay, records, revision})});
  const result = await response.json();
  if (!response.ok) throw Error(result.error);
  revision=result.revision; overlay.records=result.records; reportProgress();
  $("saveStatus").textContent="已保存独立审核文件："+result.review_file;
}
$("reviewForm").onsubmit=async e => {
  e.preventDefault();
  try {
    const record=structuredClone(overlay.records.find(r => r.review_id === current.review_id) || current.human_review);
    Object.assign(record, {reviewer:$("reviewer").value.trim(), observed_event_label:$("observed").value.trim(),
      event_presence_decision:$("presence").value, temporal_boundary_decision:$("boundary").value,
      evidence_quality:$("quality").value, notes:$("notes").value, reviewed_at:new Date().toISOString(),
      reviewed_frame_range:[+$("reviewStart").value,+$("reviewEnd").value],
      video_review_attested:$("attest").checked, review_status:"human_reviewed", human_verified:false});
    const phases=["onset", "apex", "offset"];
    record.suggested_boundaries=phases.some(p => $(p).value !== "") ? Object.fromEntries(phases.map(p => [p+"_frame", $(p).value === "" ? null : +$(p).value])) : null;
    const records=overlay.records.filter(r => r.review_id !== record.review_id).concat([record]);
    await save(records);
  } catch(e) {fail(e);}
};
$("scope").onchange=fillQueue; $("shotFilter").onchange=fillQueue;
$("queue").onchange=() => choose().catch(fail); $("feature").onchange=drawCurve;
$("previous").onclick=() => seekFrame(+$("frame").value-1); $("next").onclick=() => seekFrame(+$("frame").value+1);
$("go").onclick=() => seekFrame(+$("frame").value);
$("play").onclick=() => {windowEnd=null; $("still").hidden=true; $("video").hidden=false; $("video").play().catch(fail);};
$("window").onclick=() => {
  const q=current.source_queue_item;
  const begin=frames.frames.find(f => f.global_frame_index === +q.start_frame);
  const last=frames.frames.find(f => f.global_frame_index === +q.end_frame-1);
  const target=frames.frames.find(f => f.movie_timestamp_sec >= begin.movie_timestamp_sec-0.5) || frames.frames[0];
  seekFrame(target.global_frame_index, false);
  const after=frames.frames.find(f => f.movie_timestamp_sec >= last.next_movie_timestamp_sec+0.5);
  windowEnd=after ? after.preview_time_sec : frames.frames.length/frames.preview_fps;
  $("video").play().catch(fail);
};
function tick(time) {
  if (!frames) return;
  if (!$("video").paused) labelFrame(frameAtPreview(time));
  if (windowEnd !== null && time >= windowEnd) {$("video").pause(); windowEnd=null;}
}
if ($("video").requestVideoFrameCallback) {
  const callback=(_, meta) => {tick(meta.mediaTime); $("video").requestVideoFrameCallback(callback);};
  $("video").requestVideoFrameCallback(callback);
} else $("video").ontimeupdate=() => tick($("video").currentTime);
$("video").onplay=() => {$("still").hidden=true; $("video").hidden=false;};
$("video").onseeked=() => {if ($("still").hidden && frames) labelFrame(frameAtPreview($("video").currentTime));};
$("timeline").onclick=e => {
  const rect=e.target.getBoundingClientRect(), x=(e.clientX-rect.left)*1000/rect.width, y=(e.clientY-rect.top)*180/rect.height;
  const types=["au_activation","eye_closure_candidate","head_orientation_change","camera_relative_gaze_change"];
  const a=frames.frames[0].global_frame_index, b=frames.frames.at(-1).global_frame_index+1;
  const n=a+(x-280)/700*(b-a), type=types[Math.floor(y/40)];
  const candidates=timeline.review_intervals.filter(r => r.event_type === type && +r.start_frame <= n && n < +r.end_frame);
  candidates.sort((l,r) => (+l.end_frame-l.start_frame)-(+r.end_frame-r.start_frame));
  if (candidates.length) {$("scope").value="all"; fillQueue(); $("queue").value=candidates[0].review_id; choose().catch(fail);}
};
$("reload").onclick=() => restore().catch(fail);
$("export").onclick=() => {
  const url=URL.createObjectURL(new Blob([pretty(overlay)], {type:"application/json"}));
  const a=document.createElement("a"); a.href=url; a.download=`reviews_${index.source_run_sha256}.json`; a.click(); URL.revokeObjectURL(url);
};
$("import").onchange=async e => {
  try {
    const payload=JSON.parse(await e.target.files[0].text());
    if (payload.source_run_sha256 !== index.source_run_sha256) throw Error("Source Hash mismatch");
    // POST validates all IDs, hashes, ranges and review fields before writing.
    await save(payload.records); fillForm();
  } catch(e) {fail(e);}
};
(async () => {
  index=await get("data/index.json");
  for (const sid of [...new Set(index.items.map(i => i.shot_id))]) $("shotFilter").add(new Option(sid,sid));
  await restore(); fillQueue();
})().catch(e => {$("status").textContent="请使用 Python Review Server 打开。"; fail(e);});
