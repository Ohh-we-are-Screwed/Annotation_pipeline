"""The tail (stages 7-9) on stage6_stereo_box boxes.

stage6_stereo_box emits Stage 6's row shape, so stages 7-9 consume it unchanged
-- except in the three places that named `stage6_cluster` for reasons that have
nothing to do with clustering: Stage 7's input directory, the module a refusal
tells the operator to run, and the provenance of the boxes that ship. The first
real stereo run refused at Stage 7 with

    STAGE 7: REFUSING TO START: <work_root>/stage6_cluster/run_manifest.json
    not found; run `python3 -m pipeline.stage6_cluster.cluster` first

with a complete stage6_stereo_box tree sitting beside it, because the wrapper
picked the producer for Stage 8 and the release (`boxes_dir`) but not for Stage
7. One helper now answers it for both, and these tests pin that they agree.
"""

from __future__ import annotations

import math
import json
import os
import re
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.common.manifest import (  # noqa: E402
    UpstreamRefusal,
    boxes_module_hint,
    boxes_source,
    require_upstream,
)

WRAPPER = os.path.join(ROOT, "scripts", "run_stages.sh")


def _wrapper_text() -> str:
    with open(WRAPPER, "r", encoding="utf-8") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# The wrapper
# ---------------------------------------------------------------------------


def test_wrapper_is_syntactically_valid():
    assert subprocess.run(["bash", "-n", WRAPPER], capture_output=True).returncode == 0


def test_stage7_and_the_box_consumers_pick_the_producer_the_same_way():
    text = _wrapper_text()
    step7 = text[text.index("    7)  acc"):text.index("    8)  acc")]
    assert '--stage6-dir "$(boxes_dir_for_stage6)"' in step7, step7
    # ...and boxes_dir() (Stage 8 + the release) delegates rather than
    # re-implementing the choice, so the two can never disagree.
    boxes_dir = text[text.index("boxes_dir() {"):]
    boxes_dir = boxes_dir[:boxes_dir.index("\n}")]
    assert "boxes_dir_for_stage6" in boxes_dir
    assert "stage6_stereo_box" not in boxes_dir, "the stage6 choice lives in ONE helper"


@pytest.mark.parametrize(
    "stereo_marker,cluster_marker,stereo_newer,expected",
    [
        (True, False, False, "stage6_stereo_box"),   # only stereo ran
        (False, True, False, "stage6_cluster"),      # only cluster ran
        (True, True, True, "stage6_stereo_box"),     # both, stereo is fresher
        (True, True, False, "stage6_cluster"),       # both, stereo is STALE
        (False, False, False, "stage6_cluster"),     # neither: the default refusal
    ],
)
def test_boxes_dir_for_stage6_freshness_rule(tmp_path, stereo_marker, cluster_marker, stereo_newer, expected):
    """The helper itself, run by bash with the wrapper's own marker_state."""
    for name, marked, mtime in (
        ("stage6_cluster", cluster_marker, 1_700_000_100),
        ("stage6_stereo_box", stereo_marker, 1_700_000_200 if stereo_newer else 1_700_000_000),
    ):
        d = tmp_path / name
        d.mkdir()
        (d / "run_manifest.json").write_text("{}")
        os.utime(d / "run_manifest.json", (mtime, mtime))
        if marked:
            (d / "_SUCCESS").write_text("fingerprint\n")

    text = _wrapper_text()
    helper = text[text.index("boxes_dir_for_stage6() {"):]
    helper = helper[:helper.index("\n}") + 2]
    marker_state = text[text.index("marker_state() {"):]
    marker_state = marker_state[:marker_state.index("\n}") + 2]
    script = f'WORK_ROOT="{tmp_path}"\n{marker_state}\n{helper}\nboxes_dir_for_stage6\n'
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)
    assert os.path.basename(out.stdout.strip()) == expected


# ---------------------------------------------------------------------------
# The stages
# ---------------------------------------------------------------------------


def _stage6_tree(tmp_path, stage: str, fingerprint: str = "f" * 8):
    d = tmp_path / stage
    d.mkdir()
    (d / "run_manifest.json").write_text(json.dumps({
        "spec": f"dhakascenes-pilot/{stage}/v1",
        "stage": stage,
        "upstream": {"metadata_fingerprint": fingerprint, "fingerprint_spec": "sha256/v1"},
    }))
    (d / "_SUCCESS").write_text(fingerprint + "\n")
    return str(d)


def test_stage7_accepts_a_stereo_box_tree(tmp_path):
    """The gate is about completion and substrate, never about which producer."""
    d = _stage6_tree(tmp_path, "stage6_stereo_box")
    manifest, marker = require_upstream(
        d, stage_name="Stage 6", module_hint=boxes_module_hint(d), current_fingerprint="f" * 8,
    )
    assert manifest["spec"] == "dhakascenes-pilot/stage6_stereo_box/v1"
    assert marker.state == "clean"


def test_a_missing_tree_names_the_producer_that_was_asked_for(tmp_path):
    """"run stage6_cluster first" is wrong advice on a stereo run."""
    with pytest.raises(UpstreamRefusal) as exc:
        require_upstream(
            str(tmp_path / "stage6_stereo_box"), stage_name="Stage 6",
            module_hint=boxes_module_hint(str(tmp_path / "stage6_stereo_box")),
        )
    assert "pipeline.stage6_stereo_box.stereo_box" in str(exc.value)
    assert boxes_module_hint("/w/stage6_cluster") == "pipeline.stage6_cluster.cluster"
    assert boxes_module_hint("/w/stage7_track/") == "pipeline.stage7_track.track"


def test_boxes_source_survives_the_whole_tail():
    """Stage 9's manifest has to name the Stage 6 producer, three stages later."""
    s6 = {"stage": "stage6_stereo_box", "spec": "dhakascenes-pilot/stage6_stereo_box/v1"}
    s7 = {"stage": "stage7_track", "boxes_source": boxes_source(s6)}
    s8 = {"stage": "stage8_inflate", "boxes_source": boxes_source(s7)}
    assert boxes_source(s8) == "stage6_stereo_box"
    assert boxes_source({"stage": "stage6_cluster"}) == "stage6_cluster"
    # A pre-change manifest says "unknown" rather than naming the stage that
    # merely passed the boxes on.
    assert boxes_source({"stage": "stage7_track"}) == "unknown"


def test_every_tail_stage_records_the_producer():
    for rel in ("pipeline/stage7_track/track.py", "pipeline/stage8_inflate/inflate.py",
                "pipeline/stage9_qa/gate.py"):
        with open(os.path.join(ROOT, rel), "r", encoding="utf-8") as fh:
            text = fh.read()
        assert re.search(r'"boxes_source": boxes_source\(', text), rel


# ---------------------------------------------------------------------------
# Stage 7's cost on stereo-dense instances
#
# stage6_cluster's DBSCAN clusters are "tens-to-a-few-hundred points"; a
# stage6_stereo_box instance carries the painted STEREO returns — median 435,
# p99 23k, max 29k on chunk_0010. Two hot spots followed from that, and both
# are about faithfulness as much as speed.
# ---------------------------------------------------------------------------


def _icp_brute_force(source_xyz, target_xyz, cfg):
    """`icp_register` as it was, with the N x M distance matrix it replaced."""
    import numpy as np

    src = source_xyz.astype(np.float64).copy()
    init_t = target_xyz.mean(axis=0) - src.mean(axis=0)
    src = src + init_t
    R_total, t_total = np.eye(3), init_t.copy()
    prev_mean, n_iter, converged = float("inf"), 0, False
    for n_iter in range(1, cfg.icp_max_iterations + 1):
        d = np.linalg.norm(src[:, None, :] - target_xyz[None, :, :], axis=2)
        nn = np.argmin(d, axis=1)
        corr = target_xyz[nn]
        mean_dist = float(d[np.arange(src.shape[0]), nn].mean())
        src_c, tgt_c = src.mean(axis=0), corr.mean(axis=0)
        U, _, Vt = np.linalg.svd((src - src_c).T @ (corr - tgt_c))
        D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(Vt.T @ U.T))) or 1.0])
        R_step = Vt.T @ D @ U.T
        t_step = tgt_c - R_step @ src_c
        src = (R_step @ src.T).T + t_step
        R_total = R_step @ R_total
        t_total = R_step @ t_total + t_step
        if abs(prev_mean - mean_dist) < cfg.icp_convergence_tol_m:
            prev_mean, converged = mean_dist, True
            break
        prev_mean = mean_dist
    return {"rotation": R_total, "translation_m": t_total, "n_iterations": n_iter,
            "converged": converged, "mean_residual_m": round(prev_mean, 6),
            "n_source_points": int(source_xyz.shape[0]), "n_target_points": int(target_xyz.shape[0])}


def _assert_same_icp(got, want):
    import numpy as np

    assert np.allclose(got["rotation"], want["rotation"], atol=1e-9)
    assert np.allclose(got["translation_m"], want["translation_m"], atol=1e-9)
    assert got["mean_residual_m"] == want["mean_residual_m"]
    assert got["n_iterations"] == want["n_iterations"]
    assert got["converged"] == want["converged"]
    assert (got["n_source_points"], got["n_target_points"]) == (want["n_source_points"], want["n_target_points"])


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_icp_on_the_kdtree_matches_the_brute_force_formulation(seed):
    """Rotated, noisy, shuffled: residual > 0 and the loop actually iterates."""
    import numpy as np
    from pipeline.stage7_track.track import TrackConfig, icp_register

    rng = np.random.default_rng(seed)
    cfg = TrackConfig()
    src = rng.normal(scale=1.5, size=(240, 3))
    th = math.radians(25.0)
    R = np.array([[math.cos(th), -math.sin(th), 0.0], [math.sin(th), math.cos(th), 0.0], [0.0, 0.0, 1.0]])
    tgt = (R @ src.T).T + np.array([0.7, -0.4, 0.12]) + rng.normal(scale=0.05, size=src.shape)
    tgt = tgt[rng.permutation(tgt.shape[0])]

    want = _icp_brute_force(src, tgt, cfg)
    # the test is only worth running if the case is a real one
    assert want["mean_residual_m"] > 0.0 and want["n_iterations"] > 2, want
    _assert_same_icp(icp_register(src, tgt, cfg), want)


def test_icp_ties_do_not_change_the_result():
    """Exact duplicates in the target: a tie the tree may break differently.

    argmin takes the lowest index, cKDTree need not — but duplicates sit at the
    SAME coordinates, so the correspondence, and the transform, are identical.
    """
    import numpy as np
    from pipeline.stage7_track.track import TrackConfig, icp_register

    rng = np.random.default_rng(11)
    cfg = TrackConfig()
    src = rng.normal(scale=1.0, size=(80, 3))
    tgt = np.repeat(src + np.array([0.35, -0.2, 0.05]), 3, axis=0)  # every target point 3x
    tgt = np.vstack([tgt, np.zeros((4, 3))])  # and 4 coincident points at the origin
    _assert_same_icp(icp_register(src, tgt, cfg), _icp_brute_force(src, tgt, cfg))


def test_an_unclustered_row_is_not_re_clustered():
    """stage6_stereo_box kept no cluster, so there is nothing to replay.

    DBSCAN here would hand ICP a cluster the box was never fitted to — and
    take ~6 s on a 23k-point instance.
    """
    import numpy as np
    from pipeline.stage6_cluster.cluster import canonical_order
    from pipeline.stage7_track.track import CloudCache, reconstruct_cluster_points

    rng = np.random.default_rng(3)
    # one tight blob plus a far-away satellite: DBSCAN would drop the satellite
    cloud = np.vstack([rng.normal(scale=0.2, size=(60, 3)), rng.normal(loc=40.0, scale=0.2, size=(8, 3))])
    point_index = np.arange(cloud.shape[0], dtype=np.int64)
    instance_id = np.zeros(cloud.shape[0], dtype=np.int64)
    cache = CloudCache()
    cache._entries["kf"] = (cloud, point_index, instance_id, None)

    kept = reconstruct_cluster_points("kf", "", "", 0, 0.5, 5, cache, clustered=False)
    assert kept is not None and kept.shape[0] == cloud.shape[0]
    assert np.array_equal(kept, cloud[canonical_order(cloud, point_index)])

    clustered = reconstruct_cluster_points("kf", "", "", 0, 0.5, 5, cache, clustered=True)
    assert clustered is not None and clustered.shape[0] < cloud.shape[0]


def test_both_reconstruct_call_sites_ask_the_row():
    with open(os.path.join(ROOT, "pipeline/stage7_track/track.py"), "r", encoding="utf-8") as fh:
        text = fh.read()
    assert text.count('clustered=det.get("cluster") is not None') == 2
    assert "src[:, None, :] - target_xyz[None, :, :]" not in text, "the 12.7 GB temporary is back"
