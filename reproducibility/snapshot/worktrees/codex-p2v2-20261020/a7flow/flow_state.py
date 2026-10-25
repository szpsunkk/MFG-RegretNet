"""Frozen conditional continuation for A7, separate from original next/ sources."""
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
sys.path.insert(0, str(WORK / 'a7'))
sys.path.insert(0, str(WORK / 'next'))
import state as a7
from state import ROOT, OUT, PAPER, OPS, read, digest, atomic_json, canonical_immutable
from v2_model import FAMILIES

GAMMA = {'QV-g1': 1., 'QV-sel': 1., 'QV-g0.5': .5, 'QV-g0': 0.}
TABLE = OUT / 'A7_tau/table.json'
NAMES = ('flow_state.py', 'tau.py', 'model.py', 'check_sel.py', 'run_phase.py', 'continue_a7.py', 'test_flow.py', 'resource_worker.py')


def candidate(arm):
    if arm not in GAMMA:
        raise ValueError(arm)
    return OUT / 'continuation_A7' / arm


def seal():
    a7.check()
    files = [HERE / n for n in NAMES] + sorted((WORK / 'next').glob('*.py'))
    canonical_immutable(OUT / 'A7_continuation_manifest.json', {
        'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in files},
        'A7_manifest_sha256': digest(OUT / 'A7_manifest.json'),
    })
    check()


def check():
    a7.check()
    registry = read(OUT / 'A7_continuation_manifest.json')
    if registry['A7_manifest_sha256'] != digest(OUT / 'A7_manifest.json'):
        raise RuntimeError('A7 launch manifest changed')
    for relative, expected in registry['source_sha256'].items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError('continuation source changed: ' + relative)


def initialize(arm):
    check()
    base = candidate(arm)
    decision = read(OUT / 'qv_decision_A7.json')
    if arm not in decision['eligible_ranked']:
        raise RuntimeError('proxy not eligible under A7')
    canonical_immutable(base / 'qv_decision.json', decision)
    canonical_immutable(base / 'selection.json', {'proxy': arm, 'gamma': GAMMA[arm],
        'decision_sha256': digest(OUT / 'qv_decision_A7.json'),
        'continuation_manifest_sha256': digest(OUT / 'A7_continuation_manifest.json')})
    canonical_immutable(base / 'validation/reference_tests.json', read(OUT / 'validation/reference_tests.json'))
    if arm != 'QV-sel':
        for family in FAMILIES:
            name = f"validation/v2/g{GAMMA[arm]:g}/{family}/checks.json"
            original = read(OUT / name)
            assert original['pass_checks'] and original['model_source_sha256'] == digest(WORK / 'next/v2_model.py')
            canonical_immutable(base / name, original)
    return base


def progress_note(status):
    path = OUT / 'status_A7.json'
    snapshot = read(path) if path.exists() else {}
    preparation = OPS / 'preparation_status.json'
    prep = read(preparation) if preparation.exists() else {}
    lines = ['# A7 执行进度', '', time.strftime('%Y-%m-%d %H:%M:%S %z'), '',
             '全部运行安排在 101；本机 LN 不变。A7 优先，Stage 3 比较组低优先级且封存。', '',
             f'后续控制状态：{status}',
             f'隐私标定已检查：{prep.get("checked", 0)}/48；可运行 {prep.get("runnable", 0)}；不可达 {prep.get("excluded", 0)}。', '',
             '| 阶段 | 已完成 | 已准备任务 |', '|---|---:|---:|']
    for name, counts in snapshot.get('counts', {}).items():
        lines.append(f'| {name} | {counts["complete"]} | {counts["total"]} |')
    lines += ['', f'运行中任务数：{len(snapshot.get("active", []))}。',
              '只读取状态、轮次、耗时及完成文件存在性；全部 A7 运行完成前不解析其精度。',
              '原 QV 判定及被替换的旧运行保留。A7 判定、逐单元差值随后写入 RESULTS_A7_zh.md。']
    (PAPER / 'A7_PROGRESS_zh.md').write_text('\n'.join(lines) + '\n')
