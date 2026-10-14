"""CPU-only contract checks. No Isaac import, real torch, GPU, or PT loading."""
import contextlib
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import evaluation as ev


def tensor(value):
    class FakeTensor(np.ndarray):
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return np.asarray(self)
        def clone(self): return tensor(self.copy())
        def to(self, device): return self
    return np.asarray(value).view(FakeTensor)


def args(phase="validation", scenarios="G1,G2,G6"):
    p = ev.PHASES[phase]
    return SimpleNamespace(phase=phase, scenarios=scenarios, num_envs=p["num_envs"],
        seeds=",".join(map(str, p["seeds"])), startup_seed=p["startup_seed"],
        terrain_seed=p["terrain_seed"], task="Quadruped-v0")


def audit_run(n=3, terminal=None, timeout=None, error=(.1, -.2, .3)):
    audit = ev.ErrorAudit(n)
    for step in range(ev.STEPS):
        t = np.zeros(n, bool)
        tr = np.zeros(n, bool)
        if terminal and step in terminal:
            t[terminal[step]] = True
        if timeout and step in timeout:
            tr[timeout[step]] = True
        audit.step(np.tile(error, (n, 1)), (0, 0, 0), t, tr)
    return audit.result()


def model_fixture(scenarios, phase="validation"):
    p = ev.PHASES[phase]
    return dict(checkpoint_sha256="a"*64, protocol=ev.protocol(args(phase)),
                actor_validation={"state_key": "actor_state_dict"}, runtime={"torch": "stub"},
                source_hashes_verified=True, diagnostics_complete=True, source_audit={"sha256": "b"*64},
                seeds={str(s): {"scenarios": {sid: {"marker": sid} for sid in scenarios}} for s in p["seeds"]})


class AuditTests(unittest.TestCase):
    def test_exact_unrounded_full_window_sums(self):
        row = audit_run()
        self.assertEqual(row["valid_samples"], 3000)
        self.assertAlmostEqual(row["sum_squared_error"]["vx"], 30)
        self.assertAlmostEqual(row["sum_absolute_error"]["yaw"], 900)
        self.assertAlmostEqual(row["rmse"]["vy"], .2)
        self.assertEqual(row["completed_control_steps"], 1000)

    def test_first_terminal_tick_and_reset_trajectory_mask(self):
        row = audit_run(terminal={10: [0], 250: [1], 700: [0]})
        self.assertEqual(row["per_episode_valid_samples"], [10, 250, 1000])
        self.assertEqual(row["first_terminal_steps"], [10, 250, None])
        self.assertEqual(row["survived"], 1)

    def test_final_timeout_is_explicit_999_samples(self):
        row = audit_run(timeout={999: [0, 1, 2]})
        self.assertEqual(row["per_episode_valid_samples"], [999]*3)
        self.assertEqual(row["first_timeout_steps"], [999]*3)
        self.assertEqual(row["survived"], 3)

    def test_early_timeout_invalidates_immediately(self):
        with self.assertRaisesRegex(ev.EvaluationInvalid, "Early timeout"):
            audit_run(timeout={998: [0]})

    def test_timeout_on_already_dead_reset_is_ignored(self):
        row = audit_run(terminal={2: [0]}, timeout={900: [0]})
        self.assertIsNone(row["first_timeout_steps"][0])
        self.assertEqual(row["per_episode_valid_samples"][0], 2)

    def test_concurrent_final_termination_is_still_a_fall(self):
        row = audit_run(terminal={999: [0]}, timeout={999: [0]})
        self.assertEqual(row["falls"], 1)
        self.assertEqual(row["first_timeout_steps"][0], 999)

    def test_zero_first_step_fall_has_no_fake_rmse(self):
        row = audit_run(terminal={0: [0]})
        self.assertIsNone(row["per_episode_metrics"][0]["rmse"]["vx"])
        self.assertEqual(row["per_episode_valid_samples"][0], 0)

    def test_nan_velocity_never_sanitized(self):
        audit = ev.ErrorAudit(1)
        with self.assertRaisesRegex(ev.EvaluationInvalid, "Nonfinite"):
            audit.step([[np.nan, 0, 0]], [0, 0, 0], [False], [False])

    def test_inf_command_never_sanitized(self):
        audit = ev.ErrorAudit(1)
        with self.assertRaisesRegex(ev.EvaluationInvalid, "Nonfinite"):
            audit.step([[0, 0, 0]], [np.inf, 0, 0], [False], [False])

    def test_nonfinite_event_flag_is_not_converted_to_fall(self):
        with self.assertRaisesRegex(ev.EvaluationInvalid, "Nonfinite"):
            ev.ErrorAudit(1).step([[0,0,0]], [0,0,0], [np.nan], [False])

    def test_no_valid_samples_invalid(self):
        with self.assertRaisesRegex(ev.EvaluationInvalid, "No valid"):
            audit_run(n=1, terminal={0:[0]})

    def test_shape_mismatch_invalid(self):
        with self.assertRaises(ev.EvaluationInvalid):
            ev.ErrorAudit(2).step([[0, 0, 0]], [0, 0, 0], [False]*2, [False]*2)

    def test_event_shape_mismatch_invalid(self):
        with self.assertRaises(ev.EvaluationInvalid):
            ev.ErrorAudit(2).step([[0, 0, 0]]*2, [0, 0, 0], [False], [False])

    def test_short_control_horizon_invalid(self):
        with self.assertRaisesRegex(ev.EvaluationInvalid, "Incomplete"):
            ev.ErrorAudit(2).result()

    def test_exact_segment_counts_include_initial_transition(self):
        row = audit_run(terminal={250: [0]})
        self.assertEqual([r["valid_samples"] for r in row["segment_error_sums"]], [750, 500, 500, 500])

    def test_per_episode_distribution_reported(self):
        row = audit_run(n=2)
        self.assertEqual(len(row["per_episode_metrics"]), 2)
        self.assertEqual(row["per_episode_error_distribution"]["yaw"]["units"], "rad/s")
        self.assertAlmostEqual(row["per_episode_error_distribution"]["vx"]["quantiles"]["0.95"], .1)


class ContractTests(unittest.TestCase):
    def test_selection_protocol(self):
        self.assertEqual(ev.validate_args(args("selection", "G1,G6")), (("G1", "G6"), (2029,), "flat"))

    def test_wrong_fresh_seed_invalid(self):
        a = args()
        a.seeds = "809,1201"
        with self.assertRaises(ev.EvaluationInvalid): ev.validate_args(a)

    def test_mixed_terrain_refused(self):
        with self.assertRaisesRegex(ev.EvaluationInvalid, "terrain"):
            ev.validate_args(args(scenarios="G1,G3"))

    def test_selection_cannot_add_holdout_scenario(self):
        with self.assertRaises(ev.EvaluationInvalid): ev.validate_args(args("selection", "G2"))

    def test_duplicate_scenario_refused(self):
        with self.assertRaises(ev.EvaluationInvalid): ev.validate_args(args(scenarios="G1,G1"))

    def test_validation_protocol_exactly_decision_rules(self):
        parent = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(parent))
        import decision_rules
        self.assertEqual(ev.protocol(args()), decision_rules.PROTOCOL)

    def test_manifest_real_bytes_hash_no_policy_load(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            ck = d/"model.pt"
            ck.write_bytes(b"not a checkpoint; do not deserialize in CPU tests")
            manifest = d/"manifest.json"
            ev.write_json(manifest, [{"name": "candidate", "checkpoint": str(ck), "sha256": ev.sha256(ck)}])
            self.assertEqual(ev.read_manifest(manifest)[0]["name"], "candidate")
            ck.write_bytes(b"changed")
            with self.assertRaisesRegex(ev.EvaluationInvalid, "SHA256"):
                ev.read_manifest(manifest)

    def test_normalization_manifest_refused(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); ck = d/"a.pt"; ck.write_bytes(b"fake")
            manifest = d/"m.json"
            ev.write_json(manifest, [{"name": "candidate", "checkpoint": str(ck), "sha256": ev.sha256(ck), "actor_observation_normalization": True}])
            with self.assertRaisesRegex(ev.EvaluationInvalid, "Normalized"):
                ev.read_manifest(manifest)

    def test_manifest_relative_checkpoint_resolves_against_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); ck=d/"references"/"model.pt"; ck.parent.mkdir(); ck.write_bytes(b"never deserialize")
            manifest=d/"selection_manifest.json"
            ev.write_json(manifest,[{"name":"C0_5500","checkpoint":"references/model.pt","sha256":ev.sha256(ck)}])
            self.assertEqual(ev.read_manifest(manifest)[0]["checkpoint"],str(ck.resolve()))

    def test_source_audit_independently_rehashes_provided_and_ppo(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); provided = {}
            for rel in ("train.py", "play.py", "go2_task/agent_cfg.py", "go2_task/env_cfg.py"):
                p=d/rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(rel.encode()); provided[rel]=ev.sha256(p)
            runtime=d/"ppo.py"; runtime.write_bytes(b"protected runtime")
            audit=d/"audit.json"; ev.write_json(audit, {"ok": True, "provided": provided, "runtime": {str(runtime): ev.sha256(runtime)}})
            self.assertEqual(ev.source_audit(audit, d)["sha256"], ev.sha256(audit))
            runtime.write_bytes(b"changed")
            with self.assertRaisesRegex(ev.EvaluationInvalid, "PPO/runtime"):
                ev.source_audit(audit, d)

    def test_json_nonfinite_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError): ev.write_json(Path(d)/"bad.json", {"bad": np.inf})

    def test_merge_rejects_duplicate_scenarios(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            for i in range(2):
                folder=d/str(i)
                ev.write_json(folder/"models.json", {"candidate": model_fixture(("G1",))})
                ev.write_json(folder/"evaluation_complete.json", {"complete": True, "phase": "validation"})
            with self.assertRaisesRegex(ev.EvaluationInvalid, "Duplicate"):
                ev.merge_results([d/"0", d/"1"], d/"merge", "validation")

    def test_merge_rejects_incomplete_coverage(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); folder=d/"flat"
            ev.write_json(folder/"models.json", {"candidate": model_fixture(("G1", "G2", "G6"))})
            ev.write_json(folder/"evaluation_complete.json", {"complete": True, "phase": "validation"})
            with self.assertRaisesRegex(ev.EvaluationInvalid, "scenario coverage"):
                ev.merge_results([folder], d/"merge", "validation")

    def test_full_merge_has_no_prediction(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); folders=[]
            for kind in ("flat", "rough", "slope", "stairs", "dr"):
                folder=d/kind; folders.append(folder)
                ids=[s for s,t in ev.TERRAINS.items() if t==kind]
                ev.write_json(folder/"models.json", {"candidate": model_fixture(ids)})
                ev.write_json(folder/"evaluation_complete.json", {"complete": True, "phase": "validation"})
            result=ev.merge_results(folders, d/"merge", "validation")
            self.assertEqual(set(result["candidate"]["seeds"]["2039"]["scenarios"]), set(ev.TERRAINS))
            self.assertIsNone(json.loads((d/"merge"/"evaluation_complete.json").read_text())["official_score_prediction"])


class ActorStubTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.path=Path(self.temp.name)/"fake.pt"
        self.path.write_bytes(b"only hash these bytes; all torch.load calls below are mocked")
        self.item={"name":"candidate","checkpoint":str(self.path),"sha256":ev.sha256(self.path)}
        self.ck={"actor_state_dict":{},"iter":1499}
        for i,(out,inp) in enumerate(((512,235),(256,512),(128,256),(12,128))):
            self.ck["actor_state_dict"][str(i*2)+".weight"]=tensor(np.zeros((out,inp),np.float32))
            self.ck["actor_state_dict"][str(i*2)+".bias"]=tensor(np.zeros(out,np.float32))
        self.fake_torch=SimpleNamespace(load=lambda *a,**k:self.ck,isfinite=np.isfinite,
            zeros=lambda shape,**kw:tensor(np.zeros(shape)),float32=np.float32,inference_mode=contextlib.nullcontext)
        self.legacy=SimpleNamespace(load_checkpoint_actor=lambda *a:tensor_actor)

    def tearDown(self): self.temp.cleanup()

    def test_actor_shape_and_finite_probe(self):
        with patch.dict(sys.modules,{"torch":self.fake_torch}):
            actor,meta=ev.load_actor(self.item,self.legacy,"stub")
            self.assertEqual(meta["layers"],[[512,235],[256,512],[128,256],[12,128]])
            self.assertEqual(meta["output_probe_shape"],[2,12])

    def test_actor_normalization_state_rejected(self):
        self.ck["actor_obs_normalizer_state_dict"]={"mean":tensor(np.zeros(235))}
        with patch.dict(sys.modules,{"torch":self.fake_torch}):
            with self.assertRaisesRegex(ev.EvaluationInvalid,"Normalization"):
                ev.load_actor(self.item,self.legacy,"stub")

    def test_empty_normalization_mapping_is_not_active_normalizer(self):
        self.ck["empirical_normalization"]={}
        with patch.dict(sys.modules,{"torch":self.fake_torch}):
            self.assertEqual(ev.load_actor(self.item,self.legacy,"stub")[1]["iteration"],1499)

    def test_actor_wrong_layer_dimensions_rejected(self):
        self.ck["actor_state_dict"]["0.weight"]=tensor(np.zeros((512,234)))
        with patch.dict(sys.modules,{"torch":self.fake_torch}):
            with self.assertRaisesRegex(ev.EvaluationInvalid,"dimensions"):
                ev.load_actor(self.item,self.legacy,"stub")

    def test_nonfinite_actor_parameter_rejected(self):
        self.ck["actor_state_dict"]["0.bias"][0]=np.inf
        with patch.dict(sys.modules,{"torch":self.fake_torch}):
            with self.assertRaisesRegex(ev.EvaluationInvalid,"Nonfinite checkpoint"):
                ev.load_actor(self.item,self.legacy,"stub")

    def test_wrong_actor_output_rejected(self):
        self.legacy.load_checkpoint_actor=lambda *a:lambda obs:tensor(np.zeros((len(obs),11)))
        with patch.dict(sys.modules,{"torch":self.fake_torch}):
            with self.assertRaisesRegex(ev.EvaluationInvalid,"output probe"):
                ev.load_actor(self.item,self.legacy,"stub")


def tensor_actor(obs):
    return tensor(np.zeros((len(obs),12)))


class StubScenarioTests(unittest.TestCase):
    def test_full_scenario_stub_preserves_cached_obs_and_saves_trace(self):
        n=64
        robot=SimpleNamespace(data=SimpleNamespace(root_lin_vel_b=tensor(np.zeros((n,3))), root_ang_vel_b=tensor(np.zeros((n,3)))))
        term=SimpleNamespace(command=(0,0,0), is_standing_env=np.zeros(n,bool))
        def set_command(term_, command): term_.command=command
        calls={"step":0,"obs_compute":0}
        def step(action):
            calls["step"]+=1
            command=term.command
            robot.data.root_lin_vel_b[:]=np.array(command)+np.array((.1,-.2,0))
            robot.data.root_ang_vel_b[:,2]=command[2]+.3
            return tensor(np.zeros((n,235))), tensor(np.ones(n)), tensor(np.zeros(n,bool)), tensor(np.zeros(n,bool)), {}
        env=SimpleNamespace(unwrapped=SimpleNamespace(num_envs=n, seed=lambda s: None), reset=lambda: (tensor(np.zeros((n,235))),{}), step=step)
        def forbidden_compute():
            calls["obs_compute"]+=1
            raise AssertionError("Extra observation sampling violates frozen protocol")
        env.unwrapped.observation_manager=SimpleNamespace(compute=forbidden_compute)
        legacy=SimpleNamespace(SCENARIOS={"G1":{"terrain":"flat","cmds":[(.3,0,0),(.6,0,0),(1,0,0),(-.5,0,0)]}},set_command=set_command)
        fake_torch=SimpleNamespace(isfinite=np.isfinite, manual_seed=lambda s: None, inference_mode=contextlib.nullcontext,
                                   stack=lambda vals,dim: tensor(np.stack(vals,axis=dim)))
        bindings=(robot,None,None,["FL_foot","FR_foot","RL_foot","RR_foot"],None,1.0)
        capture=lambda b,r,c: {field:np.zeros((4,12 if "action" in field else 3)) for field in ev.TRACE_FIELDS}
        with tempfile.TemporaryDirectory() as d, patch.dict(sys.modules,{"torch":fake_torch}), patch.object(ev,"capture_diagnostics",side_effect=capture):
            path=Path(d)/"trace.npz"
            row=ev.run_scenario(env,term,lambda obs:tensor(np.zeros((n,12))),{"policy_clip_actions":None},"G1",2029,legacy,bindings,path,"stub")
            self.assertEqual(calls,{"step":1000,"obs_compute":0})
            self.assertEqual(row["valid_samples"],64000)
            self.assertAlmostEqual(row["rmse"]["yaw"],.3)
            with np.load(path) as trace:
                self.assertEqual(trace["raw_action"].shape,(1000,4,12))
                self.assertEqual(trace["control_step"].tolist(),list(range(1000)))
                self.assertEqual(trace["command"][250].tolist(),[.6,0,0])
                self.assertTrue(trace["valid_initial_episode"].all())


if __name__=="__main__":
    unittest.main(verbosity=2)
