"""Optional, explicitly selected finalist and held-out internal evaluation."""
import argparse
import json
import traceback

import sweep_runner as s


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=['none', 'action', 'acc', 'linear'], required=True)
    p.add_argument('--saved_model', required=True)
    p.add_argument('--iterations', type=int, default=3000)
    p.add_argument('--deadline', type=float, required=True)
    args = p.parse_args()
    assert 800 < args.iterations <= 3000
    snapshot = s.protected()
    assert snapshot == json.loads((s.ROOT/'previous_source_hashes.json').read_text())
    backup = s.ROOT/'fresh_rewards_original.py'
    assert backup.exists()
    saved = json.loads((s.ROOT/'saved_models.json').read_text())
    names = ['C0_5500', args.saved_model]
    assert args.saved_model.startswith('F10_')
    models = [x for x in saved if x['name'] in names]
    assert len(models) == 2
    try:
        if args.arm != 'none':
            s.apply_rewards(args.arm)
            run_name = 'final_20261002_'+args.arm
            s.job([s.ISAAC, '-p', 'train.py', '--task', 'Quadruped-v0', '--num_envs', '6144',
                   '--seed', '42', '--max_iterations', str(args.iterations), '--run_name', run_name,
                   '--headless'], 'final_train_'+args.arm, args.iterations*6+180,
                  args.deadline, run_name)
            files = list((s.ROOT/'preserved'/run_name).glob('*/model_*.pt'))
            numeric = [x for x in files if x.stem.split('_')[-1].isdigit()]
            checkpoint = max(numeric, key=lambda x: int(x.stem.split('_')[-1]))
            assert int(checkpoint.stem.split('_')[-1]) >= args.iterations-1
            models.append(dict(name='final_'+args.arm, checkpoint=str(checkpoint.relative_to(s.ROOT)),
                               sha256=s.sha(checkpoint)))
        (s.ROOT/'heldout_models.json').write_text(json.dumps(models, indent=2))
        s.apply_rewards('control')
        labels = []
        for terrain, scenarios, cap in [('flat', 'G1,G2,G6', 1800), ('rough', 'G3', 900),
                                       ('slope', 'G4', 900), ('stairs', 'G5', 900), ('dr', 'G7', 900)]:
            label = 'heldout_'+terrain
            labels.append(label)
            s.evaluate('heldout_models.json', label, scenarios, args.deadline,
                       n=128, seeds='809,1201', terrain_seed=9049, startup_seed=303, cap=cap)
        summary = s.summarize(labels, 'heldout_ranking.json')
        assert all(len(m['scenarios']) == 7 for m in summary['models'].values())
        s.state('HELDOUT_COMPLETE', ranking=summary)
    except Exception as e:
        traceback.print_exc()
        s.state('STOPPED_REQUIRES_REVIEW', error=str(e))
        raise
    finally:
        original = s.TRAIN/'quadruped_rewards.py'
        original.write_bytes(backup.read_bytes())
        assert s.protected() == snapshot and s.sha(original) == s.sha(backup)
        (s.ROOT/'reward_restored.json').write_text(json.dumps({'restored': True, 'sha256': s.sha(original)}))
        s.bundle()


if __name__ == '__main__':
    main()
