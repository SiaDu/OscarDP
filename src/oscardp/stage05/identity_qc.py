from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def contact_sheets(root: Path, clusters, samples, pairs):
    accepted = [s for s in samples if s["status"] == "embedded"]
    rows = []
    for cluster in clusters:
        for tid in cluster["track_ids"]:
            selected = [s for s in accepted if s["track_id"] == tid]
            if selected:
                indices = np.linspace(0,len(selected)-1,min(6,len(selected)),dtype=int)
                rows.append((cluster,tid,[selected[int(i)] for i in indices]))
    canvas = Image.new("RGB", (1200,max(1,len(rows))*260), "#161a20")
    draw = ImageDraw.Draw(canvas)
    for row,(cluster,tid,selected) in enumerate(rows):
        scores = [p["cosine_similarity"] for p in cluster["supporting_pairs"]]
        score = f"cluster min pair cosine {min(scores):.4f}" if scores else "unknown / no pair support"
        draw.text((8,row*260+4), f"{cluster['cluster_id']} | {tid} | {score} | tentative", fill="white")
        for col,sample in enumerate(selected):
            image = Image.open(root/sample["aligned_crop"]).convert("RGB").resize((168,168))
            canvas.paste(image,(col*200+12,row*260+28))
            draw.text((col*200+10,row*260+202),f"{sample['shot_id']}\nframe {sample['frame_index']}\nsharp {sample['quality_metrics']['sharpness_laplacian_variance_112']:.1f}",fill="#ffe17a")
    canvas.save(root/"previews/identity_contact_sheet.jpg", quality=95)
    cluster_for = {tid:c["cluster_id"] for c in clusters for tid in c["track_ids"]}
    cross = sorted([p for p in pairs if cluster_for[p["track_a"]] != cluster_for[p["track_b"]] and p["cosine_similarity"] is not None],
                   key=lambda p:-p["cosine_similarity"])[:3]
    image = Image.new("RGB",(1000,max(1,len(cross))*255),"#161a20")
    draw = ImageDraw.Draw(image)
    for row,pair in enumerate(cross):
        draw.text((8,row*255+5),f"Closest cross-cluster pair | cosine {pair['cosine_similarity']:.4f} | {pair['decision']}",fill="white")
        for col,tid in enumerate([pair["track_a"],pair["track_b"]]):
            selected = [s for s in accepted if s["track_id"] == tid]
            for k,sample in enumerate([selected[0],selected[len(selected)//2]]):
                crop = Image.open(root/sample["aligned_crop"]).convert("RGB").resize((168,168))
                x = col*500+k*235+8
                image.paste(crop,(x,row*255+28))
                draw.text((x,row*255+200),f"{cluster_for[tid]} {tid}\nframe {sample['frame_index']}",fill="#ffe17a")
    image.save(root/"previews/cross_cluster_confusions.jpg",quality=95)
