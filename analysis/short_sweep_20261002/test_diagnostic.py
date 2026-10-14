"""Meaningful CPU contract checks for the new evaluation wrapper; not Isaac Sim."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'inputs'))
import batch_eval as d
import t_eval as base


class FakeEnv:
    def __init__(self,truncate=False):
        self.unwrapped=self; self.num_envs=4; self.device='cpu'; self.step_dt=.02
        self.term=NS(vel_command_b=torch.zeros(4,3),is_standing_env=torch.zeros(4,dtype=torch.bool))
        zeros=torch.zeros(4,3)
        self.scene={'robot':NS(data=NS(root_lin_vel_b=zeros.clone(),root_ang_vel_b=zeros.clone(),
                 root_pos_w=zeros.clone(),joint_vel=torch.zeros(4,12),joint_acc=torch.zeros(4,12),applied_torque=torch.zeros(4,12)))}
        self.observation_manager=NS(group_obs_concatenate={'policy':True},active_terms={'policy':['sensors','velocity_commands']},
                                   group_obs_term_dim={'policy':[(4,),(3,)]})
        self.cfg=NS(observations=NS(policy=NS(history_length=0,velocity_commands=NS(history_length=0))))
        self.truncate=truncate
    def seed(self,seed): pass
    def observation(self): return torch.cat((torch.ones(4,4),self.term.vel_command_b.clone()),dim=1)
    def reset(self):
        self.calls=0
        return {'policy':self.observation()},{}
    def step(self,act):
        self.calls+=1
        data=self.scene['robot'].data
        data.root_lin_vel_b[:,:2]=act[:,:2]
        data.root_ang_vel_b[:,2]=act[:,2]
        return {'policy':self.observation()},torch.zeros(4),torch.zeros(4,dtype=torch.bool),torch.full((4,),self.truncate and self.calls==40),{}


def policy(obs):
    assert torch.equal(obs[:,:4],torch.ones(4,4)), 'Unrelated observations changed'
    return torch.cat((obs[:,4:7],torch.zeros(4,9)),dim=1)


def main():
    base.SCENARIOS={'TEST':dict(w=1.,desc='mock',terrain='flat',axes=('vx','vy','yaw'),
                    cmds=[(.4,0,0),(-.5,.3,0),(0,0,.6),(0,0,0)])}
    args=NS(seeds='1',num_envs=4,seg_s=.2,transient_s=0.,cmd_scale=1.,push_mag=0.,push_period_s=1.,axis_mode='all')
    env=FakeEnv()
    legacy=base.run_scenario(env,env.term,policy,'TEST',args,'cpu')
    assert legacy['rmse']['vx']>0
    env=FakeEnv()
    sync=d.SyncPolicy(policy,env.term,d.command_slice(env))
    wrapped=d.MeasuredEnv(env,env.term,10,0)
    corrected=base.run_scenario(wrapped,env.term,sync,'TEST',args,'cpu')
    assert corrected['score']==1. and sync.changed_steps==3
    measured=wrapped.summarize(ROOT/'mock_trace.npz')
    assert all(v==0 for v in measured['full']['rmse'].values())
    env=FakeEnv(True)
    try:
        base.run_scenario(d.MeasuredEnv(env,env.term,10,0),env.term,d.SyncPolicy(policy,env.term,d.command_slice(env)),'TEST',args,'cpu')
    except RuntimeError as e:
        assert 'Unexpected timeout' in str(e)
    else: raise AssertionError('Timeout should invalidate the evaluation')
    result=dict(scope='CPU mock only',legacy_rmse=legacy['rmse'],corrected_rmse=corrected['rmse'],
                unrelated_observations_preserved=True,command_resync_steps=sync.changed_steps,unexpected_timeout_rejected=True)
    (ROOT/'local_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':main()
