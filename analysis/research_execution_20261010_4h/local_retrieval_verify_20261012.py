"""Offline S5 retrieval verification. Never imports Isaac, starts a GPU or contacts a server.

Run main only after the archive has actually arrived. --self-test uses temporary
synthetic tar/NPZ fixtures and does not run main or change the execution ledger.
The sole output report is local_retrieval_verification_20261012.json. Failed
extractions are preserved for inspection; no existing file or directory is erased.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent
BASENAME = 'research_execution_20261010_4h'
REMOTE = '/workspace/student_workspace/'+BASENAME+'/'
REPORT = ROOT/'local_retrieval_verification_20261012.json'
EVIDENCE = ROOT/'live_evidence_20261012/phase_evidence_20261012_0301.json'
AXES = ('vx','vy','yaw')
PHASES = {
    'selection':dict(models=('C0_5500','F10_3000','S5_torque_half_1000','S5_torque_half_1499'),
                     seeds=('2029',),scenarios=('G1','G3','G5','G6'),envs=64,startup=404,terrain=10037,rows=16),
    'validation':dict(models=('C0_5500','F10_3000','candidate'),seeds=('2039','2063'),
                      scenarios=('G1','G2','G3','G4','G5','G6','G7'),envs=128,startup=409,terrain=11003,rows=42)
}
MECHANIC_SHAPES = {
    'projected_gravity':(1000,4,3),'body_quaternion_w':(1000,4,4),
    'foot_force_w':(1000,4,4,3),'foot_contact':(1000,4,4),
    'joint_position':(1000,4,12),'joint_velocity':(1000,4,12),'applied_torque':(1000,4,12),
    'body_linear_velocity':(1000,4,3),'body_angular_velocity':(1000,4,3),
    'world_linear_velocity':(1000,4,3),'world_angular_velocity':(1000,4,3),
    'raw_action':(1000,4,12),'clipped_action':(1000,4,12),'processed_joint_action':(1000,4,12),
    'command':(1000,3),'valid_initial_episode':(1000,4),'control_step':(1000,)
}


class VerificationError(RuntimeError):
    pass


def require(condition,label):
    if not condition:
        raise VerificationError(label)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    def invalid(value):
        raise VerificationError('Nonfinite JSON literal: '+value)
    with Path(path).open('r',encoding='utf-8-sig') as stream:
        return json.load(stream,parse_constant=invalid)


def write_report(value):
    tmp=REPORT.with_name(REPORT.name+'.tmp')
    with tmp.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write('\n')
    os.replace(tmp,REPORT)


def safe_name(name,root=BASENAME):
    require(isinstance(name,str) and name and '\\' not in name and '\x00' not in name,
            'Unsafe member name/backslash: '+repr(name))
    parts=name.split('/')
    require(not name.startswith('/') and all(p not in ('','.','..') for p in parts),'Absolute/dot/traversal member: '+name)
    require(len(parts)>1 and parts[0]==root,'Archive root basename mismatch: '+name)
    for part in parts:
        require(':' not in part and not part.endswith((' ','.')),'Windows drive/ADS/normalization member: '+name)
        require(not any(ch in '<>"|?*' for ch in part),'Windows invalid member characters: '+name)
        require(not re.fullmatch(r'(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?',part,re.I),'Windows reserved device member: '+name)
        require(not any(ord(ch)<32 for ch in part),'Control character member: '+name)
    return PurePosixPath(*parts)


def validate_tar_members(members,root=BASENAME):
    seen=set();prefix_spelling={};total=0
    require(len(members)<=20000,'Unexpected excessive member count')
    for member in members:
        safe_name(member.name,root)
        folded=member.name.casefold()
        require(folded not in seen,'Duplicate/case-colliding member: '+member.name)
        seen.add(folded)
        pieces=member.name.split('/')
        for i in range(1,len(pieces)):
            prefix='/'.join(pieces[:i]);key=prefix.casefold()
            require(prefix_spelling.get(key,prefix)==prefix,'Case-colliding implicit directory: '+prefix)
            prefix_spelling[key]=prefix
        require(member.isreg() and not member.issym() and not member.islnk() and not member.sparse,
                'Links/directories/sparse/special files rejected: '+member.name)
        require(0<=member.size<=1024*1024*1024,'Invalid/excessive member size: '+member.name)
        total+=member.size
    require(not set(prefix_spelling)&seen,'File/directory prefix collision in archive')
    require(total<=4*1024*1024*1024,'Archive expanded size exceeds4GiB safety cap')
    return total


def safe_extract(archive,destination,root=BASENAME):
    destination=Path(destination)
    require(not destination.exists(),'Refuse existing extraction target: '+str(destination))
    with tarfile.open(archive,'r:gz') as tf:
        members=tf.getmembers()
        total=validate_tar_members(members,root)
        # All names/types are inspected before even creating the destination.
        require(shutil.disk_usage(destination.parent).free>total+64*1024*1024,'Insufficient extraction disk space plus64MiB reserve')
        destination.mkdir(parents=False,exist_ok=False)
        resolved=destination.resolve()
        for member in members:
            target=destination.joinpath(*safe_name(member.name,root).parts)
            require(resolved in target.resolve().parents,'Resolved extraction path escaped target')
            target.parent.mkdir(parents=True,exist_ok=True)
            require(not target.exists() and not target.is_symlink(),'Refuse existing extraction file: '+str(target))
            with tf.extractfile(member) as source,target.open('xb') as output:
                copied=0
                while True:
                    block=source.read(1024*1024)
                    if not block:break
                    copied+=len(block)
                    require(copied<=member.size,'Tar member expands beyond declared size')
                    output.write(block)
                require(copied==member.size,'Tar member truncated: '+member.name)
    return dict(members=len(members),expanded_bytes=total,extracted_root=str(destination/root))


def manifest_verify(extracted):
    manifest=load_json(extracted/'manifest.json')
    require(manifest.get('kind')=='full' and manifest.get('contains_complete_validation') is True,
            'Full manifest does not establish complete validation')
    records=manifest.get('files')
    require(isinstance(records,dict) and records,'Missing full file manifest')
    actual={p.relative_to(extracted).as_posix() for p in extracted.rglob('*') if p.is_file()}
    require(actual-{'manifest.json'}==set(records),'Full manifest file inventory differs from extracted members')
    folded=set()
    for relative,meta in records.items():
        safe_name(BASENAME+'/'+relative)
        require(relative.casefold() not in folded,'Case-colliding manifest entry')
        folded.add(relative.casefold())
        path=extracted.joinpath(*PurePosixPath(relative).parts)
        require(path.stat().st_size==meta['bytes'],'Manifest byte mismatch: '+relative)
        require(sha(path)==meta['sha256'],'Manifest SHA mismatch: '+relative)
    return manifest,dict(files=len(records),all_bytes_and_sha256_verified=True)


def remote_relative(remote):
    require(isinstance(remote,str) and remote.startswith(REMOTE),'Trace/checkpoint path outside frozen remote root')
    relative=remote[len(REMOTE):]
    safe_name(BASENAME+'/'+relative)
    return relative


def close_values(actual,expected,label):
    import numpy as np
    require(np.allclose(actual,expected,rtol=1e-9,atol=1e-9,equal_nan=False),'Numerical reconciliation failed: '+label)


def numeric_axes(value):
    require(set(value)==set(AXES),'Missing/extra error axis')
    values=[value[a] for a in AXES]
    require(all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and v>=0 for v in values),
            'Invalid SSE/SAE values')
    return values


def verify_npz(path,row,envs,policy_clip=None):
    import numpy as np
    shapes=dict(MECHANIC_SHAPES,tracking_error_full_envs=(1000,envs,3),tracking_valid_initial_episode=(1000,envs))
    with zipfile.ZipFile(path) as zf:
        names=zf.namelist()
        require(len(names)==len(set(n.casefold() for n in names)),'Duplicate NPZ internal array')
        require(set(names)=={key+'.npy' for key in shapes},'NPZ array inventory mismatch')
        require(sum(item.file_size for item in zf.infolist())<=64*1024*1024,'Excessive expanded NPZ size')
    with np.load(path,allow_pickle=False) as archive:
        arrays={key:archive[key] for key in shapes}
    for key,value in arrays.items():
        require(value.shape==shapes[key],'NPZ shape mismatch: '+key+': '+str(value.shape))
        require(value.dtype.kind in 'biuf' and np.isfinite(value).all(),'Nonfinite/object NPZ array: '+key)
    errors=arrays['tracking_error_full_envs'];mask=arrays['tracking_valid_initial_episode']
    require(errors.dtype==np.float64,'Full tracking error dtype must befloat64')
    require(mask.dtype==np.bool_ and arrays['valid_initial_episode'].dtype==np.bool_,'NPZ masks must bebool')
    require(np.array_equal(arrays['control_step'],np.arange(1000)),'Control-step sequence incomplete')
    require(np.array_equal(arrays['valid_initial_episode'],mask[:,:4]),'Detailed/full environment mask mismatch')
    commands=np.asarray(row['segment_commands'],dtype=np.float64)
    require(commands.shape==(4,3) and np.isfinite(commands).all(),'Invalid four command segments')
    require(np.array_equal(arrays['command'],np.repeat(commands,250,axis=0)),'Trace commands differ from frozen JSON segments')
    actual=np.stack((arrays['body_linear_velocity'][:,:,0],arrays['body_linear_velocity'][:,:,1],arrays['body_angular_velocity'][:,:,2]),axis=-1)
    close_values(actual-arrays['command'][:,None,:],errors[:,:4,:],'first4 mechanical velocities minus command')
    if policy_clip is None:
        require(np.array_equal(arrays['raw_action'],arrays['clipped_action']),'Unexpected clipping in unclipped policy trace')
    require(row['episodes']==envs and row['expected_steps_per_episode']==1000 and row['planned_control_steps']==1000 and row['completed_control_steps']==1000,
            'Scenario episode/horizon coverage mismatch')
    counts=np.asarray(row['per_episode_valid_samples'])
    terminals=row['first_terminal_steps'];timeouts=row['first_timeout_steps']
    require(len(counts)==len(terminals)==len(timeouts)==envs,'Per-episode mask audit missing')
    cutoffs=[]
    for count,terminal,timeout in zip(counts.tolist(),terminals,timeouts):
        require(type(count)is int and 0<=count<=1000,'Invalid per-episode valid count')
        for event in (terminal,timeout):
            require(event is None or (type(event)is int and 0<=event<1000),'Invalid terminal/timeout index')
        require(timeout is None or timeout==999,'Unexpected early timeout')
        events=[x for x in (terminal,timeout) if x is not None]
        cutoffs.append(min(events) if events else 1000)
    require(np.array_equal(mask,np.arange(1000)[:,None]<np.asarray(cutoffs)[None,:]),'Full tracking mask disagrees with first event indices')
    require(np.array_equal(mask.sum(0),counts),'Per-episode valid sample mismatch')
    n=int(mask.sum());require(n>0 and row['valid_samples']==n,'Aggregate sample count mismatch')
    survived=sum(t is None for t in terminals)
    require(row['survived']==survived and row['falls']==envs-survived and row['nonfinite_count']==0,'Survival/nonfinite audit mismatch')
    masked=np.where(mask[:,:,None],errors,0.0)
    sse=(masked**2).sum(0);sae=np.abs(masked).sum(0)
    close_values(sse.sum(0),numeric_axes(row['sum_squared_error']),'full maskedSSE')
    close_values(sae.sum(0),numeric_axes(row['sum_absolute_error']),'full maskedSAE')
    close_values(np.sqrt(sse.sum(0)/n),[row['rmse'][a] for a in AXES],'fullRMSE')
    close_values(sae.sum(0)/n,[row['mae'][a] for a in AXES],'fullMAE')
    episode_rows=row['per_episode_metrics']
    require(len(episode_rows)==envs and row['per_episode_metrics_reported']is True,'Per-episode error metrics missing')
    for i,metric in enumerate(episode_rows):
        require(metric['env_id']==i and metric['valid_samples']==int(counts[i]) and metric['first_terminal_step']==terminals[i] and metric['first_timeout_step']==timeouts[i],
                'Per-episode metric identity/mask mismatch')
        close_values(sse[i],numeric_axes(metric['sum_squared_error']),'episodeSSE')
        close_values(sae[i],numeric_axes(metric['sum_absolute_error']),'episodeSAE')
    segments=row['segment_error_sums'];require(len(segments)==4,'Segment aggregate coverage missing')
    for i,segment in enumerate(segments):
        cut=slice(i*250,(i+1)*250)
        require(segment['segment']==i and segment['valid_samples']==int(mask[cut].sum()),'Segment sample mismatch')
        close_values((masked[cut]**2).sum((0,1)),numeric_axes(segment['sum_squared_error']),'segmentSSE')
        close_values(np.abs(masked[cut]).sum((0,1)),numeric_axes(segment['sum_absolute_error']),'segmentSAE')
    return dict(arrays=len(arrays),steps=1000,episodes=envs,valid_samples=n,finite=True,
                masked_SSE_SAE_and_episode_segment_metrics_verified=True)


def verify_phases(extracted,evidence):
    trace_seen=set();result={}
    checkpoint_hashes={}
    for entry in load_json(extracted/'selection_manifest.json'):
        checkpoint_hashes[entry['name']]=entry['sha256']
    selected=evidence['state']['selected']
    checkpoint_hashes['candidate']=selected['sha256']
    for phase,cfg in PHASES.items():
        folder=extracted/'results'/phase
        receipt=load_json(folder/'evaluation_complete.json')
        require(receipt.get('complete')is True and receipt.get('phase')==phase and receipt.get('merged')is True,'Phase completion receipt mismatch: '+phase)
        require(set(receipt['models'])==set(cfg['models']) and set(receipt['scenarios'])==set(cfg['scenarios']) and {str(s) for s in receipt['seeds']}==set(cfg['seeds']),
                'Phase receipt coverage mismatch: '+phase)
        models=load_json(folder/'models.json');require(set(models)==set(cfg['models']),'Policy coverage mismatch: '+phase)
        count=0;traces=[]
        for name,model in models.items():
            require(model['checkpoint_sha256']==checkpoint_hashes[name] and model['source_hashes_verified']is True and model['diagnostics_complete']is True,
                    'Checkpoint/source/diagnostic metadata mismatch: '+name)
            actor=model['actor_validation']
            expected_iter=5500 if name=='C0_5500' else 3000 if name=='F10_3000' else 1000 if name.endswith('_1000') else 1499
            require(actor['iteration']==expected_iter and actor['layers']==[[512,235],[256,512],[128,256],[12,128]] and actor['activation']=='ELU' and actor['output_probe_shape']==[2,12] and not actor['normalization_state_keys'],
                    'Recorded actor metadata mismatch: '+name)
            require(actor['actual_checkpoint_sha256']==model['checkpoint_sha256'],'Recorded loaded checkpointSHA mismatch')
            protocol=dict(axis_mode='all',transient_seconds=0.0,command_scale=1.0,observation_noise_enabled=True,
                          num_envs=cfg['envs'],startup_seed=cfg['startup'],terrain_seed=cfg['terrain'],segment_seconds=5.0,control_step_seconds=.02)
            require(model['protocol']==protocol,'Frozen common protocol mismatch')
            require(set(model['seeds'])==set(cfg['seeds']),'Seed coverage mismatch: '+name)
            for seed in cfg['seeds']:
                rows=model['seeds'][seed]['scenarios'];require(set(rows)==set(cfg['scenarios']),'Scenario coverage mismatch: '+name)
                for sid,row in rows.items():
                    require(row['scenario']==sid and row['reset_seed']==int(seed) and row['control_step_seconds']==.02 and row['segment_seconds']==5.0,'Scenario identity/timing mismatch')
                    detailed=row['detailed_trace'];relative=remote_relative(detailed['path'])
                    require(relative.startswith('results/'+phase+'/') and relative.endswith('/'+name+'_'+seed+'_'+sid+'.npz'),'Unexpected trace location')
                    require(relative not in trace_seen,'Trace reused across distinct evaluation rows')
                    trace_seen.add(relative);path=extracted.joinpath(*PurePosixPath(relative).parts)
                    require(path.is_file() and path.stat().st_size==detailed['bytes'] and sha(path)==detailed['sha256'],'Trace byte/hash mismatch: '+relative)
                    require(row['full_tracking_trace']['path']==detailed['path'],'Full/detailed trace path differs')
                    item=verify_npz(path,row,cfg['envs'],model.get('policy_clip_actions'))
                    traces.append(dict(model=name,seed=int(seed),scenario=sid,path=relative,**item));count+=1
        require(count==cfg['rows'],'Evaluation row count mismatch')
        result[phase]=dict(rows=count,models=len(models),traces=traces,complete_verified=True)
    all_npz={p.relative_to(extracted).as_posix() for p in (extracted/'results').rglob('*.npz')}
    require(len(trace_seen)==58 and all_npz==trace_seen,'Require exactly58 unique selection/validationNPZ files')
    return result


def checkpoints_and_training(extracted,evidence):
    state=evidence['state'];result={}
    original=state['original_reward_sha256']
    require(sha(extracted/'fresh_rewards_original.py')==original,'Recovered original reward backupSHA mismatch')
    restore=load_json(extracted/'reward_restored.json')
    require(restore.get('restored')is True and restore.get('sha256')==original,'Archived restoration marker mismatch')
    require(evidence['current_reward_sha256']==original and evidence['original_reward_sha256']==original,
            'Current reward snapshot in remote evidence did not confirm original restoration')
    current_sources=load_json(extracted/'source_hashes_current.json')
    runtime=load_json(extracted/'protected_runtime_sources.json')
    audit=load_json(extracted/'current_source_audit.json')
    require(current_sources==state['source_hashes'] and audit.get('ok')is True and audit['provided']==current_sources and audit['runtime']==runtime['files'],
            'Recovered provided/PPO source protection evidence mismatch')
    expected={str(row['model']['iteration']):row['model'] for row in state['selection_ranking']}
    for iteration in ('1000','1499'):
        meta=expected[iteration];relative=meta['checkpoint'];safe_name(BASENAME+'/'+relative)
        path=extracted.joinpath(*PurePosixPath(relative).parts)
        matches=list((extracted/'preserved/S5_torque_half').glob('*/model_'+iteration+'.pt'))
        require(len(matches)==1 and matches[0]==path,'Require unique exact preserved checkpoint'+iteration)
        require(path.stat().st_size==meta['bytes'] and sha(path)==meta['sha256'],'Checkpoint byte/SHA mismatch'+iteration)
        require(meta.get('exact_stored_iteration_verified')is True and meta.get('finite_tensor_count',0)>0,'Remote exact iteration/tensor audit missing')
        result[iteration]=dict(path=relative,bytes=path.stat().st_size,sha256=meta['sha256'],iteration=int(iteration),
             metadata_basis='Byte identity to remote exact iteration and finite-tensor verification',local_tensor_deserialization=False)
    selected=load_json(extracted/'selected_checkpoint.json');require(selected==state['selected'],'Frozen selected checkpoint differs from recovered remote evidence')
    require(selected['sha256']==result['1499']['sha256'] and selected['iteration']==1499,'Unexpected selected checkpoint')
    training=state['training'];run=(extracted/result['1499']['path']).parent
    config=training['configuration']
    for filename,key in (('env.yaml','env_sha256'),('agent.yaml','agent_sha256')):
        path=run/'params'/filename;require(path.is_file() and sha(path)==config[key],'Training parameter hash mismatch: '+filename)
    events=list(run.glob('events.out.tfevents.*'));require(events and all(p.stat().st_size>0 for p in events),'Training event files missing/empty')
    log=extracted/'logs/train_S5_torque_half.log';require(log.is_file() and log.stat().st_size>0,'Training log missing')
    text=re.sub(r'\x1b\[[0-9;]*m','',log.read_text(encoding='utf-8',errors='replace'))
    pairs=[(int(i),int(total)) for i,total in re.findall(r'Learning iteration (\d+)/(\d+)',text)]
    require(pairs==[(i,1500) for i in range(1500)],'Training log does not show exact0..1499 single-run iteration coverage')
    require(training.get('iterations_complete')is True and training.get('returncode')==0 and training.get('process_group_exited')is True,'Remote training completion/cleanup evidence missing')
    summary=load_json(extracted/'summary.json');require(summary['selected']==selected and summary['complete_validation']is True and summary['official_score_prediction']is None,'Summary completion/selected mismatch')
    require(summary['models']==load_json(extracted/'results/validation/models.json'),'Summary model data differs from merged validation')
    require(load_json(extracted/'decision.json')==summary['decision'],'Decision differs from preserved summary')
    return dict(checkpoints=result,selected=selected,event_files=len(events),env_agent_hashes_verified=True,
                training_log_exact1500_iterations=True,summary_matches_validation=True,
                provided_PPO_source_evidence_consistent=True,reward_original_backup_hash_verified=True,
                reward_restore_basis='Archived marker plus independently captured current remote SHA at phase_evidence0301; no new live-server observation')


def torch_cpu_check(path,expected_sha,iteration):
    require(sha(path)==expected_sha,'Refuse deserialization before checkpoint hash verification')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    ck=torch.load(path,map_location='cpu',weights_only=False)
    require(isinstance(ck,dict) and ck.get('iter')==iteration,'Local checkpoint stored iteration mismatch')
    count=0
    def walk(value):
        nonlocal count
        if isinstance(value,torch.Tensor):
            require(value.device.type=='cpu' and bool(torch.isfinite(value).all()),'Nonfinite/nonCPU tensor')
            count+=1
        elif isinstance(value,dict):
            for item in value.values():walk(item)
        elif isinstance(value,(tuple,list)):
            for item in value:walk(item)
    walk(ck)
    require(count>0,'Checkpoint contains no tensors')
    return dict(iteration=iteration,finite_tensor_count=count,all_tensors_on_cpu=True,torch_version=torch.__version__)


def verify(archive,receipt_path=None,check_torch=False):
    started=dt.datetime.now(dt.timezone.utc).isoformat()
    report=dict(status='VERIFYING',started_utc=started,archive=str(Path(archive).resolve()),
                evidence=str(EVIDENCE),report_path=str(REPORT),allocation_off_confirmed=False,
                scope='Local retrieval integrity only; allocation shutdown remains operator responsibility')
    try:
        evidence=load_json(EVIDENCE)
        key=BASENAME+'_results.tar.gz.sha256.json'
        receipt=evidence['archive_receipts'][key]
        state_receipt=evidence['state']['full_archive']
        require(receipt==state_receipt,'Actual remote receipt differs from recorded archive state')
        require(receipt['kind']=='full' and receipt['contains_complete_validation']is True,'Authoritative receipt is not final full results')
        if receipt_path:
            local=load_json(receipt_path)
            require(local==receipt,'Local receipt differs from recovered actual remote receipt')
        report['receipt_basis']='Actual remote .sha256.json recovered in phase_evidence0301'
        report['local_receipt_file']=str(receipt_path) if receipt_path else None
        report['remote_receipt']=receipt
        archive=Path(archive).resolve();require(archive.is_file(),'Archive has not arrived locally')
        require(not archive.name.endswith(('.crdownload','.part','.tmp')),'Incomplete download extension')
        before=(archive.stat().st_size,archive.stat().st_mtime_ns)
        require(before[0]==receipt['bytes'],'Local full archive size mismatch')
        digest=sha(archive);require(digest==receipt['sha256'],'Local full archive SHA256 mismatch')
        require(before==(archive.stat().st_size,archive.stat().st_mtime_ns),'Archive changed while hashing')
        report['archive_size_sha256_verified']=True
        destination=ROOT/('recovered_20261012_'+dt.datetime.now().strftime('%H%M%S'))
        report['extraction_destination']=str(destination)
        report['extraction']=safe_extract(archive,destination)
        extracted=destination/BASENAME
        manifest,manifest_report=manifest_verify(extracted);report['manifest']=manifest_report
        report['training']=checkpoints_and_training(extracted,evidence)
        require(manifest['selected']==report['training']['selected'],'Full manifest selected model mismatch')
        report['phases']=verify_phases(extracted,evidence)
        report['NPZ_traces_verified']=58
        if check_torch:
            for iteration,meta in report['training']['checkpoints'].items():
                env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
                child=subprocess.run([sys.executable,'-X','utf8','-B',str(Path(__file__).resolve()),'--torch-child',
                         str(extracted/meta['path']),meta['sha256'],iteration],capture_output=True,text=True,encoding='utf-8',env=env,timeout=90)
                require(child.returncode==0,'Optional CPU tensor check failed: '+child.stderr[-2000:])
                meta['local_tensor_metadata']=json.loads(child.stdout);meta['local_tensor_deserialization']=True
        report['status']='FULL_LOCAL_RETRIEVAL_VERIFIED'
        report['full_archive_retrieved_verified']=True
    except Exception as exc:
        report['status']='LOCAL_RETRIEVAL_VERIFICATION_FAILED'
        report['full_archive_retrieved_verified']=False
        report['error_type']=type(exc).__name__;report['error']=str(exc)
    report['finished_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
    write_report(report)
    print(json.dumps({k:report[k] for k in ('status','full_archive_retrieved_verified','report_path')},ensure_ascii=False))
    return 0 if report['full_archive_retrieved_verified'] else 1


def self_test():
    """Temporary synthetic checks only; does not inspect or extract actual downloads."""
    import io
    import numpy as np
    checks=0
    safe_name(BASENAME+'/a/b.json');checks+=1
    for name in ('/absolute',BASENAME+'/../bad',BASENAME+'/a\\bad',BASENAME+'/C:bad',BASENAME+'/a/CON.txt','other/a',BASENAME+'/a./b'):
        try:safe_name(name)
        except VerificationError:checks+=1
        else:raise AssertionError('Unsafe path accepted: '+name)
    with tempfile.TemporaryDirectory(prefix='retrieval_selftest_',dir=ROOT) as tmp:
        base=Path(tmp);archive=base/'good.tar.gz'
        with tarfile.open(archive,'w:gz') as tf:
            info=tarfile.TarInfo(BASENAME+'/a.txt');info.size=4;tf.addfile(info,io.BytesIO(b'test'))
        target=base/'fresh';safe_extract(archive,target)
        assert (target/BASENAME/'a.txt').read_bytes()==b'test';checks+=1
        fixture=target/BASENAME
        (fixture/'manifest.json').write_text(json.dumps(dict(kind='full',contains_complete_validation=True,
          files={'a.txt':dict(bytes=4,sha256=sha(fixture/'a.txt'))})),encoding='utf-8')
        manifest_verify(fixture);checks+=1
        (fixture/'a.txt').write_bytes(b'bad!')
        try:manifest_verify(fixture)
        except VerificationError:checks+=1
        else:raise AssertionError('Manifest hash corruption accepted')
        try:safe_extract(archive,target)
        except VerificationError:checks+=1
        else:raise AssertionError('Existing extraction reused')
        for kind in ('duplicate','link','special','directory'):
            bad=base/(kind+'.tar.gz')
            with tarfile.open(bad,'w:gz') as tf:
                first=tarfile.TarInfo(BASENAME+'/x')
                if kind=='duplicate':
                    first.size=1;tf.addfile(first,io.BytesIO(b'x'));second=tarfile.TarInfo(BASENAME+'/X');second.size=1;tf.addfile(second,io.BytesIO(b'y'))
                else:
                    first.type={'link':tarfile.SYMTYPE,'special':tarfile.FIFOTYPE,'directory':tarfile.DIRTYPE}[kind]
                    first.linkname='/outside';tf.addfile(first)
            rejected=base/('reject_'+kind)
            try:safe_extract(bad,rejected)
            except VerificationError:assert not rejected.exists();checks+=1
            else:raise AssertionError('Unsafe tar accepted:'+kind)
        # Small in-memory row uses64environmentfull errors and the exact1000-step contract.
        n=64;error=np.full((1000,n,3),.125,dtype=np.float64);mask=np.ones((1000,n),bool);mask[999]=False
        arrays={key:np.zeros(shape,dtype=np.float32) for key,shape in MECHANIC_SHAPES.items()}
        arrays.update(command=np.zeros((1000,3),np.float64),control_step=np.arange(1000),
                      valid_initial_episode=mask[:,:4],tracking_error_full_envs=error,tracking_valid_initial_episode=mask)
        arrays['body_linear_velocity'][:,:,:2]=.125;arrays['body_angular_velocity'][:,:,2]=.125
        counts=mask.sum(0);sse=((error*mask[:,:,None])**2).sum(0);sae=np.abs(error*mask[:,:,None]).sum(0)
        axes=lambda a:dict(zip(AXES,a.tolist()))
        row=dict(segment_commands=[[0,0,0]]*4,episodes=n,expected_steps_per_episode=1000,planned_control_steps=1000,completed_control_steps=1000,
          per_episode_valid_samples=counts.tolist(),first_terminal_steps=[None]*n,first_timeout_steps=[999]*n,valid_samples=int(mask.sum()),survived=n,falls=0,nonfinite_count=0,
          sum_squared_error=axes(sse.sum(0)),sum_absolute_error=axes(sae.sum(0)),rmse=axes(np.sqrt(sse.sum(0)/mask.sum())),mae=axes(sae.sum(0)/mask.sum()),
          per_episode_metrics_reported=True,per_episode_metrics=[dict(env_id=i,valid_samples=999,first_terminal_step=None,first_timeout_step=999,
             sum_squared_error=axes(sse[i]),sum_absolute_error=axes(sae[i])) for i in range(n)],
          segment_error_sums=[dict(segment=i,valid_samples=int(mask[i*250:(i+1)*250].sum()),
             sum_squared_error=axes(((error[i*250:(i+1)*250]*mask[i*250:(i+1)*250,:,None])**2).sum((0,1))),
             sum_absolute_error=axes(np.abs(error[i*250:(i+1)*250]*mask[i*250:(i+1)*250,:,None]).sum((0,1)))) for i in range(4)])
        npz=base/'trace.npz';np.savez_compressed(npz,**arrays);verify_npz(npz,row,n);checks+=1
        row['sum_squared_error']['vx']+=1
        try:verify_npz(npz,row,n)
        except VerificationError:checks+=1
        else:raise AssertionError('Corrupt SSE accepted')
        row['sum_squared_error']['vx']-=1
        arrays['tracking_error_full_envs'][2,3,0]=np.nan
        np.savez_compressed(npz,**arrays)
        try:verify_npz(npz,row,n)
        except VerificationError:checks+=1
        else:raise AssertionError('Nonfinite NPZ accepted')
    print(json.dumps(dict(self_test='PASS',checks=checks,actual_archive_main_executed=False)))
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path)
    parser.add_argument('--receipt',type=Path)
    parser.add_argument('--check-torch-cpu',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--torch-child',nargs=3,metavar=('PATH','SHA','ITERATION'),help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.self_test:return self_test()
    if args.torch_child:
        path,digest,iteration=args.torch_child
        print(json.dumps(torch_cpu_check(Path(path),digest,int(iteration))))
        return 0
    if not args.archive:parser.error('Specify an arrived local archive; no automatic remote/download operation')
    return verify(args.archive,args.receipt,args.check_torch_cpu)


if __name__=='__main__':raise SystemExit(main())
