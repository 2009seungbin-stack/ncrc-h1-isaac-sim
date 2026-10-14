"""Synthetic archive-safety and numeric checks; no policies/simulator are loaded."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("retrieval_validator", HERE / "validate_retrieval.py")
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


class ValidatorChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic_validator_", dir=HERE)
        self.base = Path(self.temp.name).resolve()
        # Check the resolved recursive-cleanup target before TemporaryDirectory cleanup.
        assert self.base.is_relative_to(HERE.resolve()) and self.base != HERE.resolve()

    def tearDown(self):
        self.temp.cleanup()

    def tar(self, entries, label="fixture"):
        path = self.base / (label + ".tar.gz")
        with tarfile.open(path, "w:gz") as tf:
            for name, kind, payload in entries:
                info = tarfile.TarInfo(name)
                info.type = kind
                info.size = len(payload) if kind == tarfile.REGTYPE else 0
                if kind == tarfile.SYMTYPE:
                    info.linkname = "../outside"
                tf.addfile(info, io.BytesIO(payload) if kind == tarfile.REGTYPE else None)
        return path

    def test_unsafe_paths_links_and_aliases_rejected(self):
        r = v.ROOT_NAME
        cases = [[(r + "/../outside", tarfile.REGTYPE, b"x")],
                 [(r + "/NUL.txt", tarfile.REGTYPE, b"x")],
                 [(r + "/file:stream", tarfile.REGTYPE, b"x")],
                 [(r + "/link", tarfile.SYMTYPE, b"")],
                 [(r + "/A/x", tarfile.REGTYPE, b"x"), (r + "/a/y", tarfile.REGTYPE, b"y")],
                 [(r + "/a", tarfile.REGTYPE, b"x"), (r + "/a/b", tarfile.REGTYPE, b"y")],
                 [(r + "/x", tarfile.REGTYPE, b"x"), (r + "/X", tarfile.REGTYPE, b"y")]]
        for i, entries in enumerate(cases):
            with self.subTest(i=i), self.assertRaises(v.Invalid):
                v.archive_members(self.tar(entries, str(i)))

    def test_existing_destination_is_not_modified(self):
        destination = self.base / "existing"
        destination.mkdir()
        (destination / "sentinel.txt").write_text("keep", encoding="utf-8")
        process = subprocess.run([sys.executable, "-B", str(HERE / "validate_retrieval.py"),
                                  "--archive", str(self.base / "missing.tar.gz"),
                                  "--sidecar", str(self.base / "missing.json"),
                                  "--destination", str(destination)], capture_output=True, text=True)
        self.assertEqual(process.returncode, 2)
        self.assertEqual([p.name for p in destination.iterdir()], ["sentinel.txt"])
        self.assertEqual((destination / "sentinel.txt").read_text(), "keep")

    def test_fresh_extraction_and_manifest_hash(self):
        payload = b"preserved bytes"
        manifest = {"created_epoch": 1, "files": {"payload.bin": {"bytes": len(payload),
                    "sha256": v.hashlib.sha256(payload).hexdigest()}}}
        path = self.tar([(v.ROOT_NAME + "/payload.bin", tarfile.REGTYPE, payload),
                         (v.ROOT_NAME + "/manifest.json", tarfile.REGTYPE, json.dumps(manifest).encode())])
        entries, _ = v.archive_members(path)
        destination = self.base / "fresh"
        destination.mkdir()
        root = v.extract_fresh(path, destination, entries)
        self.assertEqual(v.verify_manifest(root)["files"], manifest["files"])
        (root / "payload.bin").write_bytes(b"corruption")
        with self.assertRaises(v.Invalid):
            v.verify_manifest(root)

    def test_json_duplicate_and_nonfinite_rejected(self):
        path = self.base / "bad.json"
        for contents in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}'):
            path.write_text(contents)
            with self.assertRaises(v.Invalid):
                v.read_json(path)

    def trace(self, name="trace.npz", invalid_object=False):
        steps, dt = 40, .5
        error = np.zeros((steps, 128, 3), dtype=np.float32)
        error[:, :, 0] = .125
        error[:, :, 1] = -.25
        error[:, :, 2] = .5
        valid = np.ones((steps, 128), dtype=bool)
        valid[30:, 7] = False
        error[30:, 7] = np.nan  # Dead samples are outside all scored masks.
        trace = np.zeros((steps, 8, 60), dtype=np.float32)
        trace[:, :, [3, 4, 8]] = error[:, :8]
        path = self.base / name
        np.savez_compressed(path, error=(np.array([object()], dtype=object) if invalid_object else error),
                            valid=valid, trace_first_envs=trace, dt=np.array(dt),
                            trace_columns=np.array(v.TRACE_COLUMNS))
        return path, error, valid

    def test_numeric_trace_windows_and_dead_samples(self):
        path, _, _ = self.trace()
        error, valid, steady, dt = v.load_trace(path, 40)
        full, stable = v.measured_metrics(error, valid), v.measured_metrics(error, steady)
        self.assertEqual(full["count"], 5110)
        self.assertEqual(stable["count"], 4599)
        self.assertEqual(dt, .5)
        self.assertEqual(int(valid[-1].sum()), 127)
        self.assertEqual(full["rmse"], {"vx": .125, "vy": .25, "yaw": .5})
        reported = dict(full, extra_rmse={"vz": 0, "wx": 0, "wy": 0})
        v.verify_metric(reported, full, "synthetic")
        reported["sum_squared"] = [0, 0, 0]
        with self.assertRaises(v.Invalid):
            v.verify_metric(reported, full, "synthetic")

    def test_object_npz_is_rejected_without_pickle_loading(self):
        path, _, _ = self.trace(invalid_object=True)
        with self.assertRaises(v.Invalid):
            v.load_trace(path, 40)

    def test_metric_numeric_strings_and_booleans_rejected(self):
        for value in ("1", True, [0, "1"], [0, False]):
            with self.subTest(value=value), self.assertRaises(v.Invalid):
                v.close_numeric(value, value, "strict JSON numeric schema")

    def test_trace_validity_cannot_revive(self):
        path, error, valid = self.trace()
        valid[35:, 7] = True
        trace = np.zeros((40, 8, 60), dtype=np.float32)
        np.savez_compressed(path, error=error, valid=valid, trace_first_envs=trace,
                            dt=np.array(.5), trace_columns=np.array(v.TRACE_COLUMNS))
        with self.assertRaises(v.Invalid):
            v.load_trace(path, 40)


if __name__ == "__main__":
    unittest.main(verbosity=2)
