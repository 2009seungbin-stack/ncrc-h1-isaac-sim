# H1 Current Training Status

- Status: training completed; final deployment artifacts exported
- Date: 2026-09-01
- Task: Humanoid-v0
- Robot: Unitree H1 minimal, 19 actions
- Framework: Isaac Lab 2.3.2 / Isaac Sim 5.1 / RSL-RL PPO
- Training initialization: scratch training, seed 42, resume=false
- Final run: 5,000 PPO iterations
- Selected checkpoint: iteration 4,600
- Selected checkpoint mean training reward: -4.584132
- Observation dimension: 256
- Action dimension: 19
- Simulation timestep: 0.005 s
- Control timestep: 0.020 s (50 Hz)
- Action scale: 0.5
- Only confirmed reward change from the initial baseline: termination_penalty -5.0 -> -200.0
- Four-seed benchmark survival rate: 0.890625
- Four-seed tracking score: 0.634862
- Four-seed proxy scenario score: 0.566122
- Four-seed falls: 7
- Additional single-environment rerun: survival 1.0, falls 0, tracking 0.729685
- Official competition score: unavailable; the recorded scenario score is a proxy
- policy.pt validation: TorchScript load succeeded; input (1,256) -> output (1,19); all outputs finite

## Required files

- env.yaml: exact environment and reward configuration used for the final policy
- policy.pt: deployable TorchScript actor
- model_best.pt: full RSL-RL checkpoint for training continuation
- policy.onnx: optional ONNX deployment export
- SHA256SUMS.txt: integrity hashes

This bundle contains only the locally trained final H1 policy. Downloaded pretrained comparison weights are excluded.
