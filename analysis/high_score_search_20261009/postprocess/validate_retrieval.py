#!/usr/bin/env python3
"""Validate an explicitly retrieved tracking screen archive; never load policies.

Requires Python 3.10+ and NumPy. All extraction and reports go to a new explicit
destination. This script imports no simulator, training, evaluator, or torch code.
"""
from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile
import unicodedata
import zipfile

import numpy as np


ROOT_NAME = "tracking_screen_20261009"
ARMS = ("S1_tracking_pair", "R1_tracking_double")
MODELS = ("C0_5500", "F10_3000") + ARMS
GROUPS = {"flat": ("G1", "G2", "G6"), "rough": ("G3",),
          "slope": ("G4",), "stairs": ("G5",), "dr": ("G7",)}
WEIGHTS = dict(zip((f"G{i}" for i in range(1, 8)), (.15, .15, .20, .15, .15, .10, .10)))
AXES = ("vx", "vy", "yaw")
BASE = dict(track_lin_vel_xy_exp=2.0, track_ang_vel_z_exp=1.5, feet_air_time=.01,
            lin_vel_z_l2=-3.0, ang_vel_xy_l2=-.08, action_rate_l2=-.01,
            dof_acc_l2=-2.5e-7, dof_torques_l2=-.0002,
            flat_orientation_l2=0.0, dof_pos_limits=0.0)
DELTAS = {ARMS[0]: dict(track_lin_vel_xy_exp=3.0, track_ang_vel_z_exp=2.25),
          ARMS[1]: dict(track_lin_vel_xy_exp=4.0, track_ang_vel_z_exp=3.0)}
COMMANDS = {
    "G1": [[.3, 0, 0], [.6, 0, 0], [1, 0, 0], [-.5, 0, 0]],
    "G2": [[0, .5, 0], [0, -.5, 0], [.5, 0, .5], [.5, 0, -.5]],
    "G3": [[.5, 0, 0], [.5, 0, .3], [.8, 0, 0], [.5, 0, -.3]],
    "G4": [[.5, 0, 0]] * 4, "G5": [[.4, 0, 0]] * 4,
    "G6": [[.5, 0, 0]] * 4,
    "G7": [[.5, 0, 0], [.5, 0, .3], [.5, 0, 0], [.5, 0, -.3]],
}
TRACE_COLUMNS = "command3,body_lin3,body_ang3,action12,joint_vel12,joint_acc12,torque12,position3"
PROTECTED = ("train.py", "play.py", "go2_task/agent_cfg.py", "go2_task/env_cfg.py")
INPUTS = ("runner.py", "batch_eval.py", "t_eval.py", "t_eval_go2_core.py",
          "source_snapshot.py", "reference_models.json", "references/C0_5500.pt",
          "references/F10_3000.pt", "protected_runtime_sources.json")
MAX_FILES = 20000
MAX_TOTAL = 32 * 1024**3
MAX_FILE = 8 * 1024**3
MAX_NPZ = 512 * 1024**2
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {
    f"{prefix}{n}" for prefix in ("COM", "LPT") for n in range(1, 10)
}
WINDOWS_RESERVED |= {"CONIN$", "CONOUT$", "CLOCK$"} | {
    prefix + n for prefix in ("COM", "LPT") for n in ("\u00b9", "\u00b2", "\u00b3")
}


class Invalid(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise Invalid(message)


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, f"{label}: invalid integer {value!r}")
    return value


def scalar(value, label, minimum=None):
    require(type(value) in (int, float) and math.isfinite(value), f"{label}: invalid numeric scalar")
    require(minimum is None or value >= minimum, f"{label}: below minimum {minimum}")
    return float(value)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_value(value, label):
    require(isinstance(value, str) and SHA_RE.fullmatch(value), f"{label}: invalid SHA-256")
    return value


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def finite_float(text):
    value = float(text)
    require(math.isfinite(value), f"Nonfinite JSON number: {text}")
    return value


def invalid_constant(text):
    raise Invalid(f"Nonfinite JSON constant: {text}")


def read_json(path):
    path = Path(path)
    require(path.stat().st_size <= 16 * 1024**2, f"JSON exceeds limit: {path.name}")
    with path.open(encoding="utf-8-sig") as stream:
        return json.load(stream, object_pairs_hook=no_duplicates,
                         parse_float=finite_float, parse_constant=invalid_constant)


def safe_parts(name, label):
    require(isinstance(name, str) and name and "\\" not in name,
            f"{label}: empty path or backslash")
    require(not name.startswith("/") and not PurePosixPath(name).is_absolute(),
            f"{label}: absolute path")
    parts = name.split("/")
    for part in parts:
        require(part not in ("", ".", ".."), f"{label}: empty/dot/traversing component")
        require(not part.endswith((".", " ")) and len(part) <= 255,
                f"{label}: Windows path alias or long component")
        require(not any(c in '<>:"\\|?*' or unicodedata.category(c) in ("Cc", "Cs") for c in part),
                f"{label}: unsafe character")
        require(part.split(".", 1)[0].upper() not in WINDOWS_RESERVED,
                f"{label}: Windows reserved component")
    return tuple(parts)


def relative_file(root, relative, label):
    path = root.joinpath(*safe_parts(relative, label))
    require(path.is_file() and not path.is_symlink(), f"{label}: missing regular file {relative}")
    require(path.resolve().is_relative_to(root.resolve()), f"{label}: escaped root")
    return path


def archive_members(archive):
    entries, aliases, file_parts, prefixes = [], set(), set(), {}
    total = 0
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf:
            require(len(entries) < MAX_FILES, "Archive has too many members")
            name = member.name.rstrip("/") if member.isdir() else member.name
            parts = safe_parts(name, f"Tar member {member.name!r}")
            require(parts[0] == ROOT_NAME, f"Unexpected tar root: {member.name}")
            require(member.isfile() or member.isdir(), f"Tar links/special members denied: {member.name}")
            require(len(parts) > 1 or member.isdir(), "Archive root must be a directory")
            alias = unicodedata.normalize("NFC", "/".join(parts)).casefold()
            require(alias not in aliases, f"Duplicate/colliding tar destination: {member.name}")
            aliases.add(alias)
            for n in range(1, len(parts) + 1):
                prefix = "/".join(parts[:n])
                prefix_alias = unicodedata.normalize("NFC", prefix).casefold()
                require(prefixes.get(prefix_alias, prefix) == prefix,
                        f"Colliding implicit tar parent: {member.name}")
                prefixes[prefix_alias] = prefix
            integer(member.size, "Tar member size")
            require(member.size <= MAX_FILE, f"Tar member exceeds limit: {member.name}")
            if member.isfile():
                file_parts.add(parts)
                total += member.size
                require(total <= MAX_TOTAL, "Archive uncompressed files exceed limit")
            else:
                require(member.size == 0, "Directory member has payload")
            entries.append((member.name, member.type, member.size, parts))
    require(entries and file_parts, "Empty archive")
    for _, _, _, parts in entries:
        require(not any(parts[:n] in file_parts for n in range(1, len(parts))),
                f"Tar file/directory prefix conflict: {'/'.join(parts)}")
    # Finish gzip parsing before writes: validate CRC/footer and limit hidden tail.
    expanded = 0
    with gzip.open(archive, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            expanded += len(block)
            require(expanded <= total + 64 * 1024**2, "Archive hidden/padding data exceeds limit")
    return entries, total


def extract_fresh(archive, destination, entries):
    require(destination.is_dir() and not any(destination.iterdir()), "Extraction destination is not fresh/empty")
    base = destination.resolve()
    with tarfile.open(archive, "r:gz") as tf:
        seen = 0
        for member in tf:
            require(seen < len(entries), "Archive changed after inspection")
            name, kind, size, parts = entries[seen]
            require((member.name, member.type, member.size) == (name, kind, size),
                    "Archive changed after inspection")
            target = base.joinpath(*parts)
            current = base
            for component in (parts if member.isdir() else parts[:-1]):
                current = current / component
                if not current.exists():
                    current.mkdir()
                require(current.is_dir() and not current.is_symlink()
                        and current.resolve().is_relative_to(base), "Unsafe extraction parent")
            if member.isfile():
                source = tf.extractfile(member)
                require(source is not None, f"Unreadable tar file: {name}")
                written = 0
                with source, target.open("xb") as output:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(block)
                        written += len(block)
                require(written == size, f"Truncated member: {name}")
            seen += 1
        require(seen == len(entries), "Archive member count changed")
    return base / ROOT_NAME


def verify_manifest(root):
    manifest = read_json(root / "manifest.json")
    scalar(manifest["created_epoch"], "Manifest created_epoch", 0)
    files = manifest["files"]
    require(isinstance(files, dict) and files, "Empty/invalid internal manifest")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    require(actual == set(files) | {"manifest.json"},
            f"Manifest file-set mismatch: missing={sorted(set(files)-actual)}, extra={sorted(actual-set(files)-{'manifest.json'})}")
    aliases = set()
    for relative, item in files.items():
        parts = safe_parts(relative, "Manifest path")
        alias = unicodedata.normalize("NFC", "/".join(parts)).casefold()
        require(alias not in aliases, "Manifest path aliases collide")
        aliases.add(alias)
        require("__pycache__" not in parts and not parts[-1].endswith((".tmp", ".copying", ".lock")),
                f"Manifest contains excluded temporary file: {relative}")
        path = relative_file(root, relative, "Manifest")
        require(set(item) == {"bytes", "sha256"}, f"Manifest entry shape: {relative}")
        require(path.stat().st_size == integer(item["bytes"], "Manifest bytes"), f"Manifest size mismatch: {relative}")
        require(digest(path) == sha_value(item["sha256"], "Manifest SHA"), f"Manifest hash mismatch: {relative}")
    return manifest


def yaml_rewards(text):
    result, active, term = {}, False, None
    for line in text.splitlines():
        if line == "rewards:":
            active = True
            continue
        if active and line and not line[0].isspace() and not line.startswith("#"):
            break
        if not active:
            continue
        match = re.fullmatch(r"  ([A-Za-z_][A-Za-z_0-9]*):(?: null)?", line)
        if match:
            term = match.group(1)
        match = re.fullmatch(r"    weight: (\S+)\s*", line)
        if match and term:
            require(term not in result, f"Duplicate YAML reward weight: {term}")
            result[term] = scalar(float(match.group(1)), "YAML reward weight")
    return result


def patched_rewards(original, weights):
    tree = ast.parse(original.decode("utf-8"))
    values = [n.value for n in tree.body if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "REWARD_WEIGHTS" for t in n.targets)]
    require(len(values) == 1 and isinstance(values[0], ast.Dict), "Original reward literal assignment missing")
    node = values[0]
    lines = original.splitlines(keepends=True)
    start = sum(map(len, lines[:node.lineno - 1])) + node.col_offset
    end = sum(map(len, lines[:node.end_lineno - 1])) + node.end_col_offset
    return original[:start] + repr(weights).encode("utf-8") + original[end:]


def verify_state_training(root):
    state = read_json(root / "state.json")
    require(state["version"] == 1 and type(state["version"]) is int, "State version mismatch")
    require(state["stage"] == "COMPLETE_RETRIEVE_AND_STOP_SERVER", "Archive is not a final completed run")
    require(state.get("cleanup_errors") == [], "Cleanup errors missing/nonempty")
    require(state.get("error") is None or isinstance(state["error"], str), "Malformed retained error evidence")
    contract = state["contract"]
    for key, expected in dict(arms=list(ARMS), iterations=1500, train_envs=6144, train_seed=42,
                              eval_envs=128, reset_seeds=[809, 1201], terrain_seed=9049, startup_seed=303).items():
        require(contract[key] == expected and type(contract[key]) is type(expected), f"Contract mismatch: {key}")
    start = scalar(contract["server_start_epoch"], "Server start", 0)
    require(scalar(contract["hard_deadline"], "Hard deadline") == start + 21600, "Allocation contract mismatch")
    compute_deadline = scalar(contract["compute_deadline"], "Compute deadline")
    require(start < compute_deadline <= start + 18900, "Compute deadline does not preserve the 45-minute retrieval reserve")
    require(set(state["arms"]) == set(ARMS), "Unexpected training arm set")
    require(set(state["evaluations"]) == {"common_" + g for g in GROUPS}, "Unexpected evaluation group set")
    for terrain, scenarios in GROUPS.items():
        item = state["evaluations"]["common_" + terrain]
        require(item["status"] == "complete" and item["scenarios"] == ",".join(scenarios), "Incomplete evaluation state")
        scalar(item["seconds"], "Evaluation seconds", 0)
        integer(item["pid"], "Historical evaluation PID", 1)
    require(set(state["input_hashes"]) == set(INPUTS), "Unexpected experiment input hash set")
    for name, expected in state["input_hashes"].items():
        require(digest(relative_file(root, name, "Input")) == sha_value(expected, "Input SHA"), f"Input hash mismatch: {name}")
    source_hashes = read_json(root / "source_hashes_current.json")
    previous = read_json(root / "previous_source_hashes_20261002.json")
    require(source_hashes == contract["source_hashes"], "Protected source snapshot/contract mismatch")
    for name in PROTECTED:
        require(sha_value(source_hashes[name], "Protected source SHA") == previous[name], f"Historical source hash mismatch: {name}")
    runtime = read_json(root / "protected_runtime_sources.json")
    require(runtime["algorithm_package"] == "rsl_rl" and runtime["files"] == contract["runtime_source_hashes"],
            "Protected runtime manifest/contract mismatch")
    require(runtime["files"], "Empty runtime hash manifest")
    for value in runtime["files"].values():
        sha_value(value, "Runtime source SHA")
    original_path = root / "fresh_rewards_original.py"
    original = original_path.read_bytes()
    original_sha = sha_value(state["original_reward_sha256"], "Original reward SHA")
    require(digest(original_path) == original_sha, "Original reward backup hash mismatch")
    restored = read_json(root / "reward_restored.json")
    require(restored["restored"] is True and restored["sha256"] == original_sha, "Archived reward restoration evidence mismatch")
    evidence, configs = {}, {}
    for arm in ARMS:
        item = state["arms"][arm]
        weights = dict(BASE, **DELTAS[arm])
        require(item["status"] == "complete" and item["expected_weights"] == weights, f"Arm status/weights mismatch: {arm}")
        require(item["run_name"] == "tracking_20261009_" + arm, f"Run name mismatch: {arm}")
        path = relative_file(root, item["checkpoint"], "Final checkpoint")
        require(path.name == "model_1499.pt" and path.is_relative_to(root / "preserved" / arm), f"Final checkpoint path mismatch: {arm}")
        require(list((root / "preserved" / arm).glob("*/model_1499.pt")) == [path], f"Ambiguous final checkpoint: {arm}")
        require(path.parent.name.endswith("_" + item["run_name"]), f"Checkpoint run directory mismatch: {arm}")
        require(digest(path) == sha_value(item["sha256"], "Final checkpoint SHA"), f"Final checkpoint hash mismatch: {arm}")
        applied = root / (arm + "_applied_rewards.py")
        require(applied.read_bytes() == patched_rewards(original, weights), f"Applied reward source differs beyond approved literal: {arm}")
        config = item["configuration"]
        env_path, agent_path = path.parent / "params/env.yaml", path.parent / "params/agent.yaml"
        require(digest(env_path) == config["env_sha256"] and digest(agent_path) == config["agent_sha256"], f"Effective config SHA mismatch: {arm}")
        env_text, agent_text = env_path.read_text(encoding="utf-8"), agent_path.read_text(encoding="utf-8")
        actual = yaml_rewards(env_text)
        require(actual == config["effective_weights"], f"Saved/effective reward weights mismatch: {arm}")
        require(all(actual.get(k) == v for k, v in weights.items()), f"Effective approved reward weights mismatch: {arm}")
        for key, value in (("seed", "42"), ("max_iterations", "1500"), ("resume", "false")):
            require(re.search(r"^" + key + ": " + value + r"\s*$", agent_text, re.M), f"Effective {key} mismatch: {arm}")
        require(re.search(r"^  num_envs: 6144\s*$", env_text, re.M), f"Effective environment count mismatch: {arm}")
        log_path = root / "logs" / ("train_" + arm + ".log")
        require(log_path.is_file() and log_path.stat().st_size > 0, f"Training log missing: {arm}")
        log_text = re.sub(r"\x1b\[[0-9;]*m", "", log_path.read_text(encoding="utf-8", errors="replace"))
        require(re.search(r"Learning iteration\s+1499/1500\b", log_text), f"Final training iteration evidence missing: {arm}")
        profile = item["training_time_profile"]
        require(integer(profile["completed_iterations"], "Completed iterations") == 1500,
                f"Training profile is incomplete: {arm}")
        require(integer(profile["samples"], "Iteration samples") >= 1500, f"Insufficient iteration timing evidence: {arm}")
        scalar(profile["recent_seconds_per_iteration"], "Recent iteration seconds", 0)
        events = sorted(path.parent.glob("events.out.tfevents.*"))
        require(events and all(p.is_file() and p.stat().st_size > 0 for p in events), f"TensorBoard event evidence missing: {arm}")
        scalar(item["seconds"], "Training seconds", 0)
        integer(item["pid"], "Historical training PID", 1)
        configs[arm] = actual
        evidence[arm] = {"checkpoint": item["checkpoint"], "sha256": item["sha256"],
                         "env_sha256": config["env_sha256"], "agent_sha256": config["agent_sha256"],
                         "effective_weights": actual, "training_seconds": item["seconds"],
                         "final_iteration_log_confirmed": True,
                         "events": [p.relative_to(root).as_posix() for p in events]}
    require(set(configs[ARMS[0]]) == set(configs[ARMS[1]]), "Candidate reward term sets differ")
    require(all(configs[ARMS[0]][k] == configs[ARMS[1]][k] for k in configs[ARMS[0]] if k not in DELTAS[ARMS[0]]),
            "Candidate rewards differ beyond the two approved tracking terms")
    reference_applied = root / "reference_applied_rewards.py"
    require(reference_applied.read_bytes() == patched_rewards(original, BASE), "Reference evaluation rewards mismatch")
    return state, evidence


def close_numeric(actual, expected, label, rtol=2e-7, atol=2e-8):
    def numeric_types(value):
        if isinstance(value, np.ndarray):
            return value.dtype.kind in "fiu"
        if isinstance(value, (list, tuple)):
            return all(numeric_types(x) for x in value)
        return type(value) in (int, float) or isinstance(value, (np.integer, np.floating))
    require(numeric_types(actual) and numeric_types(expected), f"Non-numeric or Boolean metric: {label}")
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            and np.allclose(a, b, rtol=rtol, atol=atol), f"Numeric mismatch: {label}")


def load_trace(path, steps):
    keys = {"error", "valid", "trace_first_envs", "dt", "trace_columns"}
    with zipfile.ZipFile(path) as zf:
        infos = zf.infolist()
        require(len(infos) == 5 and {i.filename for i in infos} == {k + ".npy" for k in keys}, "NPZ entries/key set mismatch")
        require(sum(i.file_size for i in infos) <= MAX_NPZ, "NPZ expanded size exceeds limit")
        for item in infos:
            require(not item.flag_bits & 1, "Encrypted NPZ denied")
            with zf.open(item) as stream:
                version = np.lib.format.read_magic(stream)
                require(version in ((1, 0), (2, 0)), "Unsupported NPZ array header")
                header_reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                                 else np.lib.format.read_array_header_2_0)
                shape, _, dtype = header_reader(stream, max_header_size=16384)
                require(not dtype.hasobject and dtype.itemsize * math.prod(shape) <= item.file_size,
                        "Object array or excessive NPZ array shape")
                key = item.filename[:-4]
                expected_shape = {"error": (steps, 128, 3), "valid": (steps, 128),
                                  "trace_first_envs": (steps, 8, 60), "dt": (), "trace_columns": ()}[key]
                require(shape == expected_shape, f"NPZ shape mismatch: {key}")
                if key == "valid":
                    require(dtype == np.dtype(bool), "NPZ valid mask must be Boolean")
                elif key == "trace_columns":
                    require(dtype.kind in "US", "NPZ column label must be a scalar string")
                else:
                    require(dtype.kind in "fiu", f"NPZ non-real-numeric array: {key}")
    with np.load(path, allow_pickle=False, max_header_size=16384) as data:
        arrays = {key: data[key] for key in keys}
    label = arrays["trace_columns"].item()
    if isinstance(label, bytes):
        label = label.decode("utf-8")
    require(label == TRACE_COLUMNS, "NPZ trace columns mismatch")
    dt = scalar(arrays["dt"].item(), "Measured dt")
    require(0 < dt <= .5 and 4 * round(5 / dt) == steps, "Measured horizon/step count mismatch")
    valid, error, trace = arrays["valid"], arrays["error"], arrays["trace_first_envs"]
    require(not np.any(valid[1:] & ~valid[:-1]), "Trace validity revives a terminated episode")
    require(np.isfinite(error[valid]).all(), "Nonfinite scored error samples")
    live_first = valid[:, :8]
    require(np.isfinite(trace[live_first]).all(), "Nonfinite live trace samples")
    physical = trace[:, :, [3, 4, 8]] - trace[:, :, :3]
    close_numeric(error[:, :8][live_first], physical[live_first], "Trace/error physical correspondence", rtol=3e-6, atol=3e-6)
    steady = valid & ((np.arange(steps) % round(5 / dt)) >= round(.5 / dt))[:, None]
    return error, valid, steady, dt


def measured_metrics(error, mask):
    z = error[mask].astype(np.float64)
    require(len(z) > 0, "No scored samples")
    sq = z * z
    sse, absolute, signed = sq.sum(0), np.abs(z).sum(0), z.sum(0)
    top_count = max(1, math.ceil(len(z) * .05))
    top_sse = np.partition(sq, len(z) - top_count, axis=0)[-top_count:].sum(0)
    return {"count": len(z), "sum_squared": sse.tolist(), "sum_absolute": absolute.tolist(),
            "sum_error": signed.tolist(), "rmse": dict(zip(AXES, np.sqrt(sse / len(z)).tolist())),
            "mae": dict(zip(AXES, (absolute / len(z)).tolist())),
            "top5_mse_share": dict(zip(AXES, (top_sse / np.maximum(sse, 1e-20)).tolist()))}


def verify_metric(reported, measured, label):
    require(integer(reported["count"], label + ".count", 1) == measured["count"], f"Sample count mismatch: {label}")
    for key in ("sum_squared", "sum_absolute", "sum_error"):
        require(isinstance(reported[key], list) and len(reported[key]) == 3, f"Three-axis metric shape: {label}.{key}")
        close_numeric(reported[key], measured[key], label + "." + key)
    for key in ("rmse", "mae", "top5_mse_share"):
        require(set(reported[key]) == set(AXES), f"Metric axis set mismatch: {label}.{key}")
        close_numeric([reported[key][a] for a in AXES], [measured[key][a] for a in AXES], label + "." + key)
    require(set(reported["extra_rmse"]) == {"vz", "wx", "wy"}, f"Extra metric axes mismatch: {label}")
    for value in reported["extra_rmse"].values():
        scalar(value, "Extra RMSE", 0)


def pooled(metrics, episodes, survived):
    count = sum(m["count"] for m in metrics)
    sse = np.sum([m["sum_squared"] for m in metrics], axis=0)
    absolute = np.sum([m["sum_absolute"] for m in metrics], axis=0)
    signed = np.sum([m["sum_error"] for m in metrics], axis=0)
    tracking = math.exp(-math.sqrt(float(sse.sum()) / count / 3))
    survival = survived / episodes
    return {"count": count, "sum_squared": sse.tolist(), "sum_absolute": absolute.tolist(),
            "sum_error": signed.tolist(), "rmse": dict(zip(AXES, np.sqrt(sse / count).tolist())),
            "mae": dict(zip(AXES, (absolute / count).tolist())),
            "mean_error": dict(zip(AXES, (signed / count).tolist())),
            "tracking": tracking, "score": survival * tracking}


def verify_evaluations(root, state):
    models = read_json(root / "models.json")
    references = read_json(root / "reference_models.json")
    require([m["name"] for m in models] == list(MODELS), "Four-model order/name mismatch")
    require([m["name"] for m in references] == list(MODELS[:2]), "Fixed reference names mismatch")
    for i, model in enumerate(models):
        path = relative_file(root, model["checkpoint"], "Model")
        require(digest(path) == sha_value(model["sha256"], "Model SHA"), f"Model hash mismatch: {model['name']}")
        if i < 2:
            require(model == references[i], "Reference model metadata mismatch")
        else:
            arm = state["arms"][model["name"]]
            require(model["checkpoint"] == arm["checkpoint"] and model["sha256"] == arm["sha256"], "Candidate model metadata mismatch")
    expected_files, result = set(), {name: {"scenarios": {}} for name in MODELS}
    common_dt = None
    rows_total = traces_total = 0
    for terrain, scenarios in GROUPS.items():
        group = root / "results" / ("common_" + terrain)
        for model in models:
            name = model["name"]
            path = group / (name + ".json")
            expected_files.add(path.relative_to(root / "results").as_posix())
            data = read_json(path)
            require(data["model"] == name and data["checkpoint_sha256"] == model["sha256"], "Result model/hash mismatch")
            require(data["version"] == "short-sweep-v2-20261002", "Result evaluator version mismatch")
            for key, expected in (("num_envs", 128), ("startup_seed", 303), ("terrain_seed", 9049)):
                require(type(data[key]) is int and data[key] == expected, f"Result configuration mismatch: {key}")
            require(scalar(data["horizon"], "Result horizon") == 20, "Result horizon must be 20 seconds")
            command_slice = data["command_slice"]
            require(isinstance(command_slice, list) and len(command_slice) == 2
                    and all(type(x) is int for x in command_slice) and command_slice[0] >= 0
                    and command_slice[1] - command_slice[0] == 3, "Command observation slice mismatch")
            require(set(data["scenarios"]) == set(scenarios), "Result scenario set mismatch")
            for sid in scenarios:
                rows = data["scenarios"][sid]
                require(isinstance(rows, list) and len(rows) == 2 and [r["seed"] for r in rows] == [809, 1201], "Ordered reset seed coverage mismatch")
                fulls, steadies, episodes, survived = [], [], 0, 0
                for row in rows:
                    seed = integer(row["seed"], "Reset seed")
                    steps = integer(row["steps"], "Measurement steps", 1)
                    require(steps <= 20000 and integer(row["episodes"], "Episodes", 1) == 128, "Measurement shape/episode mismatch")
                    alive = integer(row["survived"], "Survived")
                    require(alive <= 128, "Survived count exceeds episodes")
                    resync = integer(row["command_resync_steps"], "Command resync steps")
                    require(resync <= steps, "Command resync count exceeds steps")
                    trace_path = group / "traces" / f"{name}_{sid}_{seed}.npz"
                    expected_files.add(trace_path.relative_to(root / "results").as_posix())
                    error, valid, steady, dt = load_trace(trace_path, steps)
                    common_dt = dt if common_dt is None else common_dt
                    require(dt == common_dt, "Measured dt differs across policies/scenarios")
                    require(int(valid[-1].sum()) == alive, "Trace/result survival mismatch")
                    full, stable = measured_metrics(error, valid), measured_metrics(error, steady)
                    require(full["count"] > stable["count"] > 0, "Full/steady sample window mismatch")
                    verify_metric(row["full"], full, f"{name}/{sid}/{seed}/full")
                    verify_metric(row["steady"], stable, f"{name}/{sid}/{seed}/steady")
                    legacy = row["legacy_summary"]
                    require(legacy["episodes"] == 128 and legacy["survived"] == alive
                            and legacy["axis_mode"] == "all" and legacy["seg_s"] == 5.0
                            and legacy["segment_cmds"] == COMMANDS[sid], "Legacy scenario contract mismatch")
                    require(legacy["nan_actions"] == 0 and legacy["crashes"] == 0, "Legacy invalid-state evidence")
                    require(sum(legacy["segment_falls"]) == 128 - alive and len(legacy["segment_falls"]) == 4, "Segment fall accounting mismatch")
                    close_numeric([legacy["rmse"][a] for a in AXES], [stable["rmse"][a] for a in AXES], "Legacy steady RMSE", rtol=0, atol=5.1e-5)
                    close_numeric([legacy["mae"][a] for a in AXES], [stable["mae"][a] for a in AXES], "Legacy steady MAE", rtol=0, atol=5.1e-5)
                    fulls.append(full)
                    steadies.append(stable)
                    episodes += 128
                    survived += alive
                    rows_total += 1
                    traces_total += 1
                require(sid not in result[name]["scenarios"], "Duplicate aggregated scenario")
                full, stable = pooled(fulls, episodes, survived), pooled(steadies, episodes, survived)
                first_count = full["count"] - stable["count"]
                first_sse = np.asarray(full["sum_squared"]) - np.asarray(stable["sum_squared"])
                require(np.all(first_sse >= -1e-7), "Negative first-window squared error")
                result[name]["scenarios"][sid] = {
                    "episodes": episodes, "survived": survived, "survival": survived / episodes,
                    "full": full, "steady": stable,
                    "first_0_5s_of_each_segment": {"count": first_count, "sum_squared": np.maximum(first_sse, 0).tolist(),
                                                 "rmse": dict(zip(AXES, np.sqrt(np.maximum(first_sse, 0) / first_count).tolist()))}}
    actual_files = {p.relative_to(root / "results").as_posix() for p in (root / "results").rglob("*") if p.is_file()}
    require(actual_files == expected_files and len(expected_files) == 76, "Unexpected/missing result JSON or trace files")
    require(rows_total == traces_total == 56, "Seed-row/trace completeness mismatch")
    for name, model in result.items():
        require(set(model["scenarios"]) == set(WEIGHTS), "Aggregated G1-G7 coverage mismatch")
        model["falls"] = sum(s["episodes"] - s["survived"] for s in model["scenarios"].values())
        for window in ("full", "steady"):
            model["weighted_" + window + "_quality"] = sum(WEIGHTS[sid] * s[window]["score"] for sid, s in model["scenarios"].items())
            model["weighted_" + window + "_axis_mse"] = {a: sum(WEIGHTS[sid] * s[window]["rmse"][a]**2 for sid, s in model["scenarios"].items()) for a in AXES}
            model["weighted_" + window + "_axis_mae"] = {a: sum(WEIGHTS[sid] * s[window]["mae"][a] for sid, s in model["scenarios"].items()) for a in AXES}
    return result, common_dt


def verify_summary(root, models):
    summary = read_json(root / "summary.json")
    require(set(summary["models"]) == set(MODELS) and summary["training_seeds"] == [42], "Summary model/training seed mismatch")
    require(summary["warning"] == "Internal full-window comparison; not official points"
            and summary["known_evaluator_issue"] == "Unchanged one-step push-observation timing", "Summary warning contract mismatch")
    recomputed = {}
    for name, model in models.items():
        expected = summary["models"][name]
        require(set(expected["scenarios"]) == set(WEIGHTS) and expected["falls"] == model["falls"], "Summary coverage/falls mismatch")
        close_numeric(expected["weighted_quality"], model["weighted_full_quality"], "Summary weighted full quality")
        scenarios = {}
        for sid, s in model["scenarios"].items():
            recorded = expected["scenarios"][sid]
            require(recorded["episodes"] == s["episodes"] and recorded["survived"] == s["survived"], "Summary episode counts mismatch")
            for key in ("survival", "tracking", "score"):
                value = s["survival"] if key == "survival" else s["full"][key]
                close_numeric(recorded[key], value, "Summary " + key)
            close_numeric([recorded["rmse"][a] for a in AXES], [s["full"]["rmse"][a] for a in AXES], "Summary full RMSE")
            scenarios[sid] = {"episodes": s["episodes"], "survived": s["survived"], "survival": s["survival"],
                              "rmse": s["full"]["rmse"], "tracking": s["full"]["tracking"], "score": s["full"]["score"]}
        recomputed[name] = {"scenarios": scenarios, "weighted_quality": model["weighted_full_quality"], "falls": model["falls"]}
        if name in ARMS:
            non_regression = all(s["survival"] >= models["C0_5500"]["scenarios"][sid]["survival"] for sid, s in model["scenarios"].items())
            require(expected["survival_non_regression_vs_C0_5500"] is non_regression, "Summary survival non-regression mismatch")
            recomputed[name]["survival_non_regression_vs_C0_5500"] = non_regression
            model["survival_non_regression_vs_C0_5500"] = non_regression
    return recomputed


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def korean_report(models, archive_sha):
    lines = ["회수 파일 검증 통과: 정책 4개 × G1–G7 × 리셋 시드 2개.",
             "20개 결과 JSON, 28개 정책·시나리오, 56개 시드 결과와 56개 수치 trace를 검증했습니다.",
             "신규 모델 2개: 각 1500회, 환경 6144개, 학습 시드 42, scratch 설정 및 최종 체크포인트 SHA 확인.",
             "", "정책 | 전 구간 품질 | 0.5초 제외 품질 | 낙상/1792"]
    for name in MODELS:
        m = models[name]
        lines.append(f"{name} | {m['weighted_full_quality']:.6f} | {m['weighted_steady_quality']:.6f} | {m['falls']}/1792")
    for name in ARMS:
        verdict = "통과" if models[name]["survival_non_regression_vs_C0_5500"] else "미통과"
        lines.append(f"{name}: C0 대비 G1–G7 각 시나리오 생존 비퇴행 {verdict}.")
    lines.extend(["", "전 구간 품질은 생존 중 모든 측정 tick의 3축 제곱오차를 합쳐 계산했습니다.",
                  "MAE와 steady 지표는 analysis.json에 별도로 기록했습니다. steady는 매 5초 구간의 첫 0.5초를 제외합니다.",
                  "이 값은 내부 비교이며 공식 점수가 아닙니다. 학습 시드는 1개이고 기준 모델의 학습 길이는 서로 다릅니다.",
                  "G6 밀침 관측의 기존 1-step 시차가 유지됩니다. 압축파일은 서버 종료 여부를 증명하지 않습니다.",
                  "archive SHA-256: " + archive_sha])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--sidecar", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path,
                        help="Nonexistent extraction/report directory; its parent must already exist")
    args = parser.parse_args()
    destination = args.destination.absolute()
    created = False
    try:
        require(not destination.exists() and not destination.is_symlink(), "Extraction destination already exists")
        require(destination.parent.is_dir(), "Extraction destination parent does not exist")
        archive, sidecar = args.archive.resolve(strict=True), args.sidecar.resolve(strict=True)
        require(archive.is_file() and sidecar.is_file() and archive != sidecar, "Invalid archive/sidecar paths")
        metadata = read_json(sidecar)
        require(isinstance(metadata, dict) and set(metadata) == {"path", "bytes", "sha256"}
                and isinstance(metadata["path"], str) and metadata["path"], "Archive sidecar schema mismatch")
        archive_sha = sha_value(metadata["sha256"], "Archive sidecar SHA")
        archive_size = integer(metadata["bytes"], "Archive sidecar bytes", 1)
        require(archive.stat().st_size == archive_size, "Archive/sidecar byte count mismatch")
        require(digest(archive) == archive_sha, "Archive/sidecar SHA-256 mismatch")
        entries, uncompressed_size = archive_members(archive)
        destination.mkdir(parents=False, exist_ok=False)
        created = True
        root = extract_fresh(archive, destination, entries)
        require(archive.stat().st_size == archive_size and digest(archive) == archive_sha,
                "Archive changed during extraction")
        manifest = verify_manifest(root)
        state, evidence = verify_state_training(root)
        models, measured_dt = verify_evaluations(root, state)
        recomputed = verify_summary(root, models)
        comparisons = {}
        for arm in ARMS:
            comparisons[arm] = {}
            for reference in MODELS[:2]:
                delta = models[arm]["weighted_full_quality"] - models[reference]["weighted_full_quality"]
                comparisons[arm][reference] = {"weighted_full_quality_delta": delta,
                                               "weighted_steady_quality_delta": models[arm]["weighted_steady_quality"] - models[reference]["weighted_steady_quality"],
                                               "fall_count_delta": models[arm]["falls"] - models[reference]["falls"]}
        validation = {"status": "PASS", "archive": str(archive), "archive_bytes": archive_size,
                      "archive_sha256": archive_sha, "sidecar": str(sidecar),
                      "sidecar_remote_path_metadata": metadata.get("path"), "extraction_root": str(root),
                      "manifest_created_epoch": manifest["created_epoch"], "verified_manifest_files": len(manifest["files"]),
                      "uncompressed_file_bytes": uncompressed_size, "state_stage": state["stage"],
                      "cleanup_errors": state["cleanup_errors"], "training_evidence": evidence,
                      "retained_historical_error": state.get("error"),
                      "result_json_count": 20, "model_scenario_count": 28, "seed_row_count": 56,
                      "numeric_trace_count": 56, "measured_dt": measured_dt,
                      "checkpoint_deserialization": False, "simulator_or_policy_execution": False,
                      "server_shutdown_verified": False,
                      "notes": ["Runtime source hashes are archived snapshot evidence; live server sources were not read.",
                                "TensorBoard events are required nonempty files and hash verified; they are not parsed.",
                                "Nonfinite post-termination samples are allowed outside the scored masks."]}
        analysis = {"status": "VALIDATED_OFFLINE_RECOMPUTATION", "models": models,
                    "recomputed_full_window_summary_models": recomputed, "candidate_comparisons": comparisons,
                    "weights": WEIGHTS, "training_seeds": [42], "reset_seeds": [809, 1201],
                    "metric_definition": "Pool per-axis SSE/count across reset seeds; tracking=exp(-sqrt(mean(axis MSE))); quality=sum(weight*survival*tracking).",
                    "steady_definition": "Exclude first 0.5 seconds of every 5-second segment, including initial and repeated-command segments.",
                    "known_evaluator_issue": "Unchanged one-step push-observation timing",
                    "comparison_limit": "Internal measurements; no official-point prediction, independent training seed repeats, or matched-duration causal controls."}
        write_json(destination / "validation.json", validation)
        write_json(destination / "analysis.json", analysis)
        report = korean_report(models, archive_sha)
        (destination / "report_ko.txt").write_text(report, encoding="utf-8", newline="\n")
        print(report, end="")
        print(json.dumps({"validation": str(destination / "validation.json"),
                          "analysis": str(destination / "analysis.json"),
                          "report": str(destination / "report_ko.txt")}, ensure_ascii=False))
        return 0
    except Exception as exc:
        failure = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}",
                   "destination": str(destination), "extraction_may_be_partial": destination.exists(),
                   "no_policy_execution": True}
        if created:
            # Keep failed extraction as evidence; never delete or reuse it automatically.
            try:
                failure_path = destination / ("validation_failure.json" if (destination / "validation.json").exists() else "validation.json")
                write_json(failure_path, failure)
                (destination / "report_ko.txt").write_text("회수 파일 검증 실패: " + str(exc) + "\n", encoding="utf-8")
            except OSError:
                pass
        print(json.dumps(failure, ensure_ascii=False), file=sys.stderr)
        print("회수 파일 검증 실패. 출력 폴더가 생성됐다면 증거로 남기며, 다음 실행은 새 폴더를 지정해야 합니다.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
