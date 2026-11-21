import numpy as np

from oscardp.stage06_2.iris_comparison import COLORS, mark_jumps, measure


def test_normalization_and_occlusion_use_same_local_basis():
    lids = (np.array([[0,0],[10,-3],[20,0]]),np.array([[0,0],[10,3],[20,0]]))
    centre = np.array([12.,1.])
    iris = centre + [[1,0],[0,1],[-1,0],[0,-1]]
    angle = .2
    rotation = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    transform = lambda a: 2*np.asarray(a)@rotation.T+[300,140]
    row = measure(transform(centre),transform(iris),tuple(map(transform,lids)))
    np.testing.assert_allclose(row["xy"],[.6,.05])
    assert measure(centre,iris,lids,blocked="closure_candidate")["xy"] is None
    assert measure([12,8],iris,lids)["status"] == "centre_outside_lids"
    narrow = tuple(v*[1,.1] for v in lids)
    assert measure([12,0],iris,narrow)["status"] == "low_visibility"


def test_jumps_do_not_bridge_windows_missing_frames_or_invalid_eyes():
    def row(n,window,x):
        return {"frame":n,"window":window,"time_s":n*.04,"next_s":(n+1)*.04,
                **{eye:{model:{"xy":[x,0] if x is not None else None,"jump":False}
                        for model in COLORS} for eye in ("image_left","image_right")}}
    records=[row(0,"a",.5),row(1,"a",.6),row(2,"b",.8),row(4,"b",.6),row(5,"b",None),row(6,"b",.8)]
    mark_jumps(records)
    assert [r["image_left"]["mediapipe"]["jump"] for r in records] == [False,True,False,False,False,False]
