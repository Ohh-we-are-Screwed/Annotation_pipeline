"""Stage 0 — what the probe is asked to look at, and what it admits it skipped.

Two costs met on the full-fused capture, 2026-09-08. The probe ran 11,204 s and
then hard-stopped, because:

  - it probes every scene in scene.json. The chain was invoked for ONE chunk
    (`--scenes dhaka_20260905_174950_chunk_0000`) and Stage 0 took no such
    argument, so it walked all eleven;
  - `check_files` stats and parses every sample_data row, and this export ships
    ~40,000 SWEEP rows per scene against ~9,500 keyframe rows. Under the dhaka6
    profile the accumulation window is the anchor keyframe alone
    (w_acc_count 1, w_acc_duration_ns 0) — nothing downstream ever opens those
    sweeps, so 81 % of a 10-20 minute spinning-disk walk bought nothing.

Scoping a check is only honest if the report says the scope changed. These
tests hold that line: a skipped sweep is COUNTED and NAMED as skipped, exactly
the way the file already distinguishes "RADAR was measured, not gated" from
"RADAR was fine".
"""

from __future__ import annotations

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.common.paths import METADATA_TABLES, load_paths  # noqa: E402
from pipeline.common.schemas import SUBSTRATE_PROFILES  # noqa: E402
from pipeline.stage0_data_probe import probe as P  # noqa: E402


# ---------------------------------------------------------------------------
# Which rows the substrate will actually read
# ---------------------------------------------------------------------------


class TestSweepsAreRead:
    """The accumulation window decides, not the directory a file sits in."""

    def test_a_window_of_several_sweeps_reads_them(self):
        assert P.sweeps_are_read(5, 500_000_000) is True

    def test_the_anchor_alone_reads_no_sweep(self):
        assert P.sweeps_are_read(1, 0) is False

    def test_a_duration_that_reaches_back_reads_sweeps_even_at_count_one(self):
        # count is the gate's expectation, duration is the reach. A window with
        # any reach can contain a non-keyframe row, so it must be checked.
        assert P.sweeps_are_read(1, 500_000_000) is True

    def test_a_count_above_one_reads_sweeps_even_at_zero_duration(self):
        assert P.sweeps_are_read(5, 0) is True

    @pytest.mark.parametrize("profile", ["dhaka", "nuscenes"])
    def test_the_sweep_reading_profiles_still_check_sweep_files(self, profile):
        p = SUBSTRATE_PROFILES[profile]
        assert P.sweeps_are_read(p["w_acc_count"], p["w_acc_duration_ns"]) is True

    def test_dhaka6_declares_the_anchor_alone_and_so_skips_sweep_files(self):
        p = SUBSTRATE_PROFILES["dhaka6"]
        assert P.sweeps_are_read(p["w_acc_count"], p["w_acc_duration_ns"]) is False

    def test_the_default_arguments_are_the_active_profiles(self):
        from pipeline.common.schemas import W_ACC_COUNT, W_ACC_DURATION_NS

        assert P.sweeps_are_read() == P.sweeps_are_read(W_ACC_COUNT, W_ACC_DURATION_NS)


# ---------------------------------------------------------------------------
# check_files scope
# ---------------------------------------------------------------------------


def _substrate(tmp_path, rows) -> P.Substrate:
    """A Substrate carrying just what check_files reads: the per-scene rows,
    the calibrated-sensor -> channel map, and a dataroot to resolve against."""
    paths = P.Paths(
        dataroot=str(tmp_path),
        meta_root=str(tmp_path),
        version="v1.0-x",
        work_root=str(tmp_path / "w"),
        out_root=str(tmp_path / "o"),
        probe_out_root=str(tmp_path / "p"),
    )
    return P.Substrate(
        paths=paths,
        tables={},
        channel_of_calibrated_sensor={"cs_lidar": "LIDAR_TOP", "cs_radar": "RADAR_FRONT"},
        sample_data_by_scene={"sc": rows},
        samples_by_token={},
        sample_data_by_token={},
        ego_pose_tokens=set(),
        calibrated_sensor_tokens=set(),
        instance_by_token={},
        category_tokens=set(),
        annotations_by_sample={},
    )


def _cloud(tmp_path, name: str, points: int | None = None) -> str:
    points = P.PCD_MIN_POINTS if points is None else points
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * (points * P.POINT_RECORD_BYTES))
    return name


def _row(filename: str, *, key: bool, cs: str = "cs_lidar") -> dict:
    return {
        "token": filename,
        "sample_token": "s1",
        "calibrated_sensor_token": cs,
        "ego_pose_token": "ep",
        "timestamp": 1000,
        "is_key_frame": key,
        "fileformat": "pcd",
        "filename": filename,
    }


class TestCheckFilesScope:
    def _rows(self, tmp_path, *, sweep_ok: bool):
        _cloud(tmp_path, "samples/LIDAR_TOP/000001.pcd.bin")
        if sweep_ok:
            _cloud(tmp_path, "sweeps/LIDAR_TOP/000001.pcd.bin")
        return [
            _row("samples/LIDAR_TOP/000001.pcd.bin", key=True),
            _row("sweeps/LIDAR_TOP/000001.pcd.bin", key=False),
        ]

    def test_a_sweeps_reading_profile_still_fails_on_a_missing_sweep(self, tmp_path):
        sub = _substrate(tmp_path, self._rows(tmp_path, sweep_ok=False))
        resolve, _, _ = P.check_files(sub, {"token": "sc"}, include_sweeps=True)
        assert resolve.ok is False
        assert resolve.detail["n_checked"] == 2
        assert resolve.detail["first_missing"] == ["sweeps/LIDAR_TOP/000001.pcd.bin"]

    def test_an_anchor_only_profile_does_not_open_the_sweep_at_all(self, tmp_path):
        sub = _substrate(tmp_path, self._rows(tmp_path, sweep_ok=False))
        resolve, parse, _ = P.check_files(sub, {"token": "sc"}, include_sweeps=False)
        assert resolve.ok is True
        assert parse.ok is True
        assert resolve.detail["n_checked"] == 1

    def test_the_skipped_sweeps_are_counted_not_silently_dropped(self, tmp_path):
        sub = _substrate(tmp_path, self._rows(tmp_path, sweep_ok=False))
        resolve, _, _ = P.check_files(sub, {"token": "sc"}, include_sweeps=False)
        # "we did not look" must be readable off the report, and distinguishable
        # from "we looked and it was fine".
        assert resolve.detail["sweeps_checked"] is False
        assert resolve.detail["n_sweep_rows_not_checked"] == 1
        assert "sweep" in resolve.detail["scope"].lower()

    def test_a_sweeps_reading_profile_records_that_it_did_look(self, tmp_path):
        sub = _substrate(tmp_path, self._rows(tmp_path, sweep_ok=True))
        resolve, _, _ = P.check_files(sub, {"token": "sc"}, include_sweeps=True)
        assert resolve.detail["sweeps_checked"] is True
        assert resolve.detail["n_sweep_rows_not_checked"] == 0

    def test_a_broken_keyframe_still_fails_when_sweeps_are_skipped(self, tmp_path):
        rows = self._rows(tmp_path, sweep_ok=True)
        (tmp_path / "samples/LIDAR_TOP/000001.pcd.bin").unlink()
        sub = _substrate(tmp_path, rows)
        resolve, _, _ = P.check_files(sub, {"token": "sc"}, include_sweeps=False)
        assert resolve.ok is False
        assert resolve.detail["first_missing"] == ["samples/LIDAR_TOP/000001.pcd.bin"]

    def test_an_unparseable_keyframe_still_fails_when_sweeps_are_skipped(self, tmp_path):
        rows = self._rows(tmp_path, sweep_ok=True)
        _cloud(tmp_path, "samples/LIDAR_TOP/000001.pcd.bin", points=P.PCD_MIN_POINTS - 1)
        sub = _substrate(tmp_path, rows)
        _, parse, _ = P.check_files(sub, {"token": "sc"}, include_sweeps=False)
        assert parse.ok is False
        assert parse.detail["n_unparseable"] == 1

    def test_a_non_required_channel_is_still_measured_and_still_not_gated(self, tmp_path):
        rows = self._rows(tmp_path, sweep_ok=True)
        rows.append(_row("samples/RADAR_FRONT/000001.pcd", key=True, cs="cs_radar"))
        sub = _substrate(tmp_path, rows)
        resolve, _, measurement = P.check_files(sub, {"token": "sc"}, include_sweeps=False)
        assert resolve.ok is True
        assert measurement["n_non_required_files"] == 1
        assert measurement["n_non_required_missing"] == 1

    def test_a_skipped_sweep_is_not_counted_as_a_non_required_file(self, tmp_path):
        # The two exemptions are different kinds and must not be pooled: RADAR is
        # out of scope by channel, a sweep is out of scope by window.
        sub = _substrate(tmp_path, self._rows(tmp_path, sweep_ok=False))
        _, _, measurement = P.check_files(sub, {"token": "sc"}, include_sweeps=False)
        assert measurement["n_non_required_files"] == 0

    def test_the_default_follows_the_active_profile(self, tmp_path):
        sub = _substrate(tmp_path, self._rows(tmp_path, sweep_ok=True))
        default, _, _ = P.check_files(sub, {"token": "sc"})
        pinned, _, _ = P.check_files(sub, {"token": "sc"}, include_sweeps=P.sweeps_are_read())
        assert default.detail == pinned.detail


# ---------------------------------------------------------------------------
# Scene selection
# ---------------------------------------------------------------------------


class TestSelectScenes:
    SCENES = [{"name": "a"}, {"name": "b"}, {"name": "c"}]

    def test_no_selection_probes_everything_and_skips_nothing(self):
        probed, skipped = P.select_scenes(self.SCENES, None)
        assert [s["name"] for s in probed] == ["a", "b", "c"]
        assert skipped == []

    def test_an_empty_selection_is_the_same_as_none(self):
        # `--scenes` with no values (nargs="*") must not mean "probe nothing".
        probed, skipped = P.select_scenes(self.SCENES, [])
        assert [s["name"] for s in probed] == ["a", "b", "c"]
        assert skipped == []

    def test_a_subset_is_probed_and_the_rest_is_named_as_skipped(self):
        probed, skipped = P.select_scenes(self.SCENES, ["b"])
        assert [s["name"] for s in probed] == ["b"]
        assert skipped == ["a", "c"]

    def test_selection_order_does_not_reorder_the_probe(self):
        probed, _ = P.select_scenes(self.SCENES, ["c", "a"])
        assert [s["name"] for s in probed] == ["a", "c"]

    def test_an_unknown_scene_name_hard_stops_and_names_it(self):
        with pytest.raises(P.HardStop) as exc:
            P.select_scenes(self.SCENES, ["b", "nope"])
        assert "nope" in str(exc.value)

    def test_a_duplicate_name_is_probed_once(self):
        probed, skipped = P.select_scenes(self.SCENES, ["b", "b"])
        assert [s["name"] for s in probed] == ["b"]
        assert skipped == ["a", "c"]


# ---------------------------------------------------------------------------
# End to end on a synthetic root
# ---------------------------------------------------------------------------


def _jpeg(path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = b"\xff\xd8\xff" + b"\0" * (size - 5) + b"\xff\xd9"
    path.write_bytes(body)


def _root(tmp_path, n_scenes: int = 2, n_keyframes: int = 3):
    """A minimal nuScenes root that passes every predicate under the ACTIVE
    substrate profile: the profile's required ring, its parse bands, and enough
    LiDAR sweeps behind each anchor to fill its accumulation window."""
    from pipeline.common.schemas import REQUIRED_CHANNELS, W_ACC_COUNT

    dataroot = tmp_path / "data"
    version = "v1.0-t"
    vdir = dataroot / version
    vdir.mkdir(parents=True)
    (dataroot / "samples").mkdir()
    (dataroot / "sweeps").mkdir()

    cams = [c for c in REQUIRED_CHANNELS if c != "LIDAR_TOP"]
    sensors = [{"token": "sen_LIDAR_TOP", "channel": "LIDAR_TOP", "modality": "lidar"}]
    sensors += [{"token": f"sen_{c}", "channel": c, "modality": "camera"} for c in cams]
    cal = [{"token": f"cs_{s['channel']}", "sensor_token": s["token"], "translation": [0, 0, 0],
            "rotation": [1, 0, 0, 0], "camera_intrinsic": []} for s in sensors]

    step_us = 500_000  # 2 Hz keyframes: every window sits inside the scene
    sweep_us = 100_000  # under MAX_SWEEP_GAP_NS (149.4 ms) at any profile
    n_sweeps = max(W_ACC_COUNT - 1, 0)

    samples, sample_data, ego, scenes, logs = [], [], [], [], []
    for si in range(n_scenes):
        scene_token = f"sc{si}"
        logs.append({"token": f"lg{si}", "location": ["boston-seaport", "singapore-onenorth"][si % 2],
                     "date_captured": "2026-09-08", "vehicle": "v", "logfile": "l"})
        chain = []
        for ki in range(n_keyframes):
            tok = f"{scene_token}_s{ki}"
            t = 1_000_000_000 + si * 100_000_000 + ki * step_us
            chain.append(tok)
            samples.append({"token": tok, "timestamp": t, "scene_token": scene_token,
                            "prev": f"{scene_token}_s{ki-1}" if ki else "",
                            "next": f"{scene_token}_s{ki+1}" if ki + 1 < n_keyframes else ""})
            ego.append({"token": f"ep_{tok}", "timestamp": t, "translation": [0.0, 0.0, 0.0],
                        "rotation": [1.0, 0.0, 0.0, 0.0]})
            name = f"{si}{ki:05d}"
            _cloud(dataroot, f"samples/LIDAR_TOP/{name}.pcd.bin")
            sample_data.append({"token": f"sd_{tok}_LIDAR_TOP", "sample_token": tok,
                                "calibrated_sensor_token": "cs_LIDAR_TOP", "ego_pose_token": f"ep_{tok}",
                                "timestamp": t, "is_key_frame": True, "fileformat": "pcd",
                                "filename": f"samples/LIDAR_TOP/{name}.pcd.bin", "prev": "", "next": ""})
            for wi in range(n_sweeps):
                fn = f"sweeps/LIDAR_TOP/{name}_{wi}.pcd.bin"
                _cloud(dataroot, fn)
                sample_data.append({"token": f"sd_{tok}_LIDAR_TOP_w{wi}", "sample_token": tok,
                                    "calibrated_sensor_token": "cs_LIDAR_TOP", "ego_pose_token": f"ep_{tok}",
                                    "timestamp": t - (n_sweeps - wi) * sweep_us, "is_key_frame": False,
                                    "fileformat": "pcd", "filename": fn, "prev": "", "next": ""})
            for c in cams:
                fn = f"samples/{c}/{name}.jpg"
                _jpeg(dataroot / fn, P.JPEG_MIN_BYTES)
                sample_data.append({"token": f"sd_{tok}_{c}", "sample_token": tok,
                                    "calibrated_sensor_token": f"cs_{c}", "ego_pose_token": f"ep_{tok}",
                                    "timestamp": t, "is_key_frame": True, "fileformat": "jpg",
                                    "filename": fn, "prev": "", "next": ""})
        scenes.append({"token": scene_token, "name": f"chunk_{si:04d}", "log_token": f"lg{si}",
                       "description": "day", "nbr_samples": n_keyframes,
                       "first_sample_token": chain[0], "last_sample_token": chain[-1]})

    tables = {name: [] for name in METADATA_TABLES}
    tables.update({"sensor.json": sensors, "calibrated_sensor.json": cal, "sample.json": samples,
                   "sample_data.json": sample_data, "ego_pose.json": ego, "scene.json": scenes,
                   "log.json": logs})
    for name, rows in tables.items():
        (vdir / name).write_text(json.dumps(rows))

    cfg = tmp_path / "paths.yaml"
    cfg.write_text(
        f"dataroot: {dataroot}\nmeta_root: {dataroot}\nversion: {version}\n"
        f"work_root: {tmp_path / 'work'}\nout_root: {tmp_path / 'out'}\n"
        f"probe_out_root: {tmp_path / 'probe'}\n"
    )
    return load_paths(str(cfg))


class TestRunProbeSceneSelection:
    def test_the_synthetic_root_passes_every_predicate(self, tmp_path):
        paths = _root(tmp_path)
        allowlist, report, _ = P.run_probe(paths, str(tmp_path / "work"))
        assert [s["name"] for s in allowlist["scenes"]] == ["chunk_0000", "chunk_0001"]
        assert report["totals"]["failing_predicate_counts"] == {n: 0 for n in P.PREDICATES}

    def test_a_scene_subset_yields_an_allowlist_of_exactly_that_subset(self, tmp_path):
        paths = _root(tmp_path)
        allowlist, _, _ = P.run_probe(paths, str(tmp_path / "work"), scene_names=["chunk_0001"])
        assert [s["name"] for s in allowlist["scenes"]] == ["chunk_0001"]
        assert allowlist["manifest"]["usable_scene_tokens"] == ["sc1"]

    def test_the_allowlist_says_which_scenes_were_never_probed(self, tmp_path):
        # Stage 1 reads this file and nothing else. An allowlist that named one
        # scene without saying the other ten were unexamined would read as
        # "ten scenes failed", which is a different and false claim.
        paths = _root(tmp_path)
        allowlist, report, _ = P.run_probe(paths, str(tmp_path / "work"), scene_names=["chunk_0001"])
        assert allowlist["scenes_not_probed"] == ["chunk_0000"]
        assert report["scenes_not_probed"] == ["chunk_0000"]
        assert report["scenes_requested"] == ["chunk_0001"]

    def test_a_full_probe_records_an_empty_skip_list_not_a_missing_one(self, tmp_path):
        paths = _root(tmp_path)
        allowlist, report, _ = P.run_probe(paths, str(tmp_path / "work"))
        assert allowlist["scenes_not_probed"] == []
        assert report["scenes_requested"] is None

    def test_totals_separate_what_was_probed_from_what_is_in_the_metadata(self, tmp_path):
        paths = _root(tmp_path)
        _, report, _ = P.run_probe(paths, str(tmp_path / "work"), scene_names=["chunk_0001"])
        totals = report["totals"]
        assert totals["n_scenes"] == 1
        assert totals["n_scenes_in_metadata"] == 2
        assert totals["n_scenes_not_probed"] == 1
        assert totals["n_excluded"] == 0

    def test_only_the_probed_scene_appears_in_the_per_scene_report(self, tmp_path):
        paths = _root(tmp_path)
        _, report, _ = P.run_probe(paths, str(tmp_path / "work"), scene_names=["chunk_0001"])
        assert [s["name"] for s in report["scenes"]] == ["chunk_0001"]

    def test_an_unknown_scene_name_hard_stops_before_any_file_is_touched(self, tmp_path):
        paths = _root(tmp_path)
        with pytest.raises(P.HardStop) as exc:
            P.run_probe(paths, str(tmp_path / "work"), scene_names=["chunk_0009"])
        assert "chunk_0009" in str(exc.value)

    def test_a_subset_that_is_wholly_unusable_still_hard_stops(self, tmp_path):
        paths = _root(tmp_path)
        os.remove(os.path.join(paths.dataroot, "samples/LIDAR_TOP/100000.pcd.bin"))
        with pytest.raises(P.HardStop):
            P.run_probe(paths, str(tmp_path / "work"), scene_names=["chunk_0001"])

    def test_the_report_records_whether_sweep_files_were_checked(self, tmp_path):
        paths = _root(tmp_path)
        _, report, _ = P.run_probe(paths, str(tmp_path / "work"))
        assert report["config"]["sweep_files_checked"] is P.sweeps_are_read()


class TestProbeCli:
    def test_scenes_is_accepted_and_reaches_run_probe(self, tmp_path, monkeypatch):
        paths = _root(tmp_path)
        seen = {}

        def fake(paths_, out_dir, scene_names=None):
            seen["scene_names"] = scene_names
            raise P.HardStop("stop here")

        monkeypatch.setattr(P, "run_probe", fake)
        code = P.main(["--paths", str(tmp_path / "paths.yaml"), "--scenes", "chunk_0001"])
        assert code == P.EXIT_HARD_STOP
        assert seen["scene_names"] == ["chunk_0001"]

    def test_no_scenes_argument_means_none_not_an_empty_probe(self, tmp_path, monkeypatch):
        _root(tmp_path)
        seen = {}

        def fake(paths_, out_dir, scene_names=None):
            seen["scene_names"] = scene_names
            raise P.HardStop("stop here")

        monkeypatch.setattr(P, "run_probe", fake)
        P.main(["--paths", str(tmp_path / "paths.yaml")])
        assert seen["scene_names"] is None
