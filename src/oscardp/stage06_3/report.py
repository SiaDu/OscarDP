"""Review queue, timeline figure and run summary for the real Pilot."""

from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter

from oscardp.stage06_2.evidence_replay import write_json
from .inference import CHARACTERS

FOCUS_217 = (1475.391,1476.058)
LEVELS = {"gaze_supported":"#206789","head_supported":"#b07b32",
          "context_only":"#7771a3","insufficient":"#c7ccd0"}
SHORT = {"MICHELLE":"M","TEDDY":"T","DON":"D","AWAY":"away","OTHER":"other","UNKNOWN":"?"}


def short_targets(record):
    names = [SHORT[t["target"]] for t in record["target_candidates"] if t["target"]!="UNKNOWN"]
    return "/".join(names) if names else "?"


def figure(output, source, records, characters):
    fig, axes = plt.subplots(3,1,figsize=(17,9),gridspec_kw={"height_ratios":[1.65,1,1]})
    fig.subplots_adjust(hspace=.67,left=.095,right=.98,top=.86,bottom=.08)
    views = [(source["sequence_start_s"],source["sequence_end_s"],"Full sequence  |  evidence and candidate targets"),
             (1474.9,1476.85,"Shot 217  |  relative eye motion review"),
             (1510.09,1528.37,"Shot 222  |  group shot")]
    fig.suptitle("BUGONIA   /   seq_006   /   gaze target candidates",fontsize=21,fontweight="bold",y=.978)
    fig.legend(handles=[Patch(facecolor=c,label=k.replace("_"," ")) for k,c in LEVELS.items()]
                       +[Patch(facecolor="white",edgecolor="black",label="outlined: named target proposal"),
                         Patch(facecolor=LEVELS["insufficient"],hatch="///",label="head only, inconclusive")],
               loc="upper center",bbox_to_anchor=(.55,.943),ncol=6,frameon=False)
    for ax,(start,end,title) in zip(axes,views,strict=True):
        names = (*CHARACTERS,"UNKNOWN")
        y = {p:i for i,p in enumerate(reversed(names))}
        for r in records:
            a,b = max(start,r["start_s"]),min(end,r["end_s"])
            if b<=a:
                continue
            selected = r["selected_target"]!="UNKNOWN"
            level = r["evidence_level"]
            ax.barh(y[r["character"]],b-a,left=a,height=.54,color=LEVELS[level],
                    edgecolor="black" if selected else "white",linewidth=.95 if selected else .24,
                    hatch="///" if r["direction_source"]=="head_only" and level=="insufficient" else None)
            if b-a > (end-start)*.022 and level!="insufficient":
                ax.text((a+b)/2,y[r["character"]],short_targets(r),ha="center",va="center",
                        fontsize=7,color="white",clip_on=True)
        for shot in source["shots"]:
            bounds = shot["stage01_temporal"]
            s,e = max(start,bounds["start"]),min(end,bounds["end"])
            if e<=s:
                continue
            if start<bounds["start"]<end:
                ax.axvline(bounds["start"],color="#666",lw=.7,ls=":",zorder=0)
            if (e-s)/(end-start)>.018:
                ax.text((s+e)/2,1.015,shot["shot_id"][-3:],ha="center",va="bottom",fontsize=8,
                        transform=ax.get_xaxis_transform())
        if start<FOCUS_217[0]<end:
            ax.axvspan(*FOCUS_217,color="#e38348",alpha=.18)
        for content in characters.values():
            for t in content["transitions"]:
                x = (t["start_s"]+t["end_s"])/2
                if start<=x<=end:
                    ax.axvline(x,color="#29383c",linewidth=.7,ls="--")
        ax.set_title(title,loc="left",pad=19,fontweight="bold",fontsize=11)
        ax.set(yticks=list(y.values()),yticklabels=list(y),xlim=(start,end),
               xlabel="Movie global time (s)")
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.grid(axis="x",lw=.4,alpha=.27)
        ax.set_axisbelow(True)
        ax.spines[["top","right","left"]].set_visible(False)
    fig.savefig(output/"gaze_target_review.png",dpi=155)
    plt.close(fig)


def publish_review(output, common, source, shot_stats, mapping, records, characters):
    review = []
    for r in records:
        focus = ((r["shot_id"]=="shot_000217" and r["start_s"]<FOCUS_217[1] and r["end_s"]>FOCUS_217[0])
                 or r["track_id"]=="shot_000222_track_002")
        if focus or (r["direction_source"] in ("gaze","head_only") and r["representative_frames"]):
            review.append({k:r[k] for k in ("interval_id","shot_id","track_id","character","start_s","end_s",
                                            "target_candidates","selected_target","evidence_level","uncertainty_reason",
                                            "representative_frames","direction_source")} | {"priority":"focus" if focus else "candidate_check"})
    write_json(output/"review_items.json",{**common,"review_items":review})
    figure(output,source,records,characters)
    by_shot = {s["shot_id"]:[r for r in records if r["shot_id"]==s["shot_id"]] for s in shot_stats}
    levels = Counter(r["evidence_level"] for r in records)
    transitions = [t for c in characters.values() for t in c["transitions"]]
    trans = Counter(t["type"] for t in transitions)
    priority = [r for r in review if r["priority"]=="focus"]
    with (output/"review.md").open("x",encoding="utf-8") as f:
        f.write("# Bugonia seq_006 · Cross-shot Gaze Target Inference\n\n"
                f"覆盖 {len(shot_stats)} 个 Shot，{len(mapping)} 条 Track，{len(records)} 个区间。"
                f"等级计数为 {dict(levels)}。人物标签和目标均为候选。\n\n"
                "| Shot | 有测量 Track 的角色候选 | Gaze / Head / Context / Insufficient |\n"
                "| --- | --- | --- |\n")
        for sid, rr in by_shot.items():
            names = "、".join(f"{m['track_id'][-3:]} {m['character']}" for m in mapping if m["shot_id"]==sid and m["measured"])
            c = Counter(r["evidence_level"] for r in rr)
            f.write(f"| {sid} | {names or '—'} | {' / '.join(str(c[k]) for k in LEVELS)} |\n")
        f.write("\n## 候选目标分布\n\n")
        for actor in CHARACTERS:
            subset = [r for r in records if r["character"]==actor]
            options = Counter(c["target"] for r in subset for c in r["target_candidates"])
            selected = Counter(r["selected_target"] for r in subset)
            f.write(f"- {actor} 的候选计数 {dict(options)}，主选计数 {dict(selected)}。\n")
        f.write("\n## Hold / Switch\n\n"
                f"Hold {trans['target_hold_candidate']} 个，Switch {trans['target_switch_candidate']} 个，"
                f"其中跨镜头 {sum(t['cross_shot'] for t in transitions)} 个。\n")
        for t in transitions:
            f.write(f"- {t['type']}，边界 {t['start_s']:.6f}–{t['end_s']:.6f} s，"
                    f"{t['from_shot_id']} → {t['to_shot_id']}，{t['from_target']} → {t['to_target']}。"
                    f"相邻稳定段覆盖 {t['from_stable_s'][0]:.3f}–{t['to_stable_s'][1]:.3f} s。\n")
        f.write("\n## 优先看原片的区间\n\n")
        groups = (
            ("217 的相对眼动", [r for r in priority if r["shot_id"]=="shot_000217"]),
            ("222 的 Michelle 侧脸", [r for r in priority if r["track_id"]=="shot_000222_track_002"]),
        )
        for title,group in groups:
            c = Counter((r["direction_source"],short_targets(r),r["selected_target"]) for r in group)
            f.write(f"- {title}涉及 {len(group)} 段，结果（方向来源、候选、主选）为 {dict(c)}。"
                    "各区间及可查帧见 review_items.json。\n")
        named = sorted((r for r in records if r["selected_target"]!="UNKNOWN"),
                       key=lambda r:r["end_s"]-r["start_s"],reverse=True)[:4]
        for r in named:
            f.write(f"- 可区分样本 {r['shot_id']} {r['character']} → {r['selected_target']}，"
                    f"{r['start_s']:.3f}–{r['end_s']:.3f} s。\n")
        f.write("\n方向匹配只比较粗略顶视平面的水平相容性。机位网格、旧机位点和名义取景方向分别保留。"
                "未标定高度、俯仰、遮挡及布局之外的对象。相似机位组和对白只提供上下文。"
                "孤立碎片和不可观测区间中止连续性，眼动只作为伴随证据。\n")
