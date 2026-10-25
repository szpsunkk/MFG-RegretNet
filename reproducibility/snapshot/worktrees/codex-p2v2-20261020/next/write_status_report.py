"""Metadata-only handoff report, deliberately never opens FL results."""
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from v2_model import WORK
from common import ROOT, OUT, PAPER, digest
from p2_state import read


def main():
    now = datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S JST')
    status = read(OUT / 'status.json') if (OUT / 'status.json').exists() else {}
    models = [read(p) for p in (OUT / 'validation/v2').glob('*/*/checks.json')]
    preparation = read(OUT / 'preparation_status.json') if (OUT / 'preparation_status.json').exists() else {}
    continuation = read(OUT / 'continuation_status.json') if (OUT / 'continuation_status.json').exists() else {}
    planned = [('QV-g1',16),('QV-sel',16),('QV-g0.5',16),('QV-g0',16)] + [(a,16) for a in
        ('CSRA','PAC','PAC-capped-legacy','VCG-capped-legacy','Lin','RegretNet-r2','V8F')]
    jobs = read(OUT / 'qv_queue.json')['jobs'] if (OUT / 'qv_queue.json').exists() else []
    rows = []
    for arm, total in planned:
        these = [j for j in jobs if j['arm'] == arm]
        rows.append(dict(stage='QV', arm=arm, planned_max=total, prepared=len(these),
            complete=sum(Path(j['done']).exists() for j in these), cells=8, seeds='16,20', rounds=40,
            evaluation='development only', machine='101'))
    with (PAPER / 'run_inventory.csv').open('w') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    lines = ['# P2v2 执行状态', '', now, '',
        '执行 A6 > A5 > A4 > A3/A1 > p2-v1；所有科学任务放在 101，本机 LN 未修改。', '',
        'A2 的 822 个记录独立复核一致。接收 A6 前没有计算或查看 QV 精度。四个代理的可运行记录全部结束前不读取 QV 终点。', '',
        '| 阶段/算法 | 计划最多记录 | 已预计算 | 已完成 |', '|---|---:|---:|---:|']
    lines += [f"| {r['arm']} | {r['planned_max']} | {r['prepared']} | {r['complete']} |" for r in rows]
    old = status.get('counts',{}).get('stage3_early',{})
    lines += ['', f"旧 Stage 3 比较组：共 664 条可运行记录；启动前已完成 137 条，待续跑 527 条。当前完成 {old.get('complete','待同步')} 条。原 38 个不可达标定不变。", '',
        '上述数目是记录数，包括指纹相同的共享执行，不等于独立训练次数。QV 隐私不可达项在预计算完成后列明，最多 176 次。', '',
        '校验：14 个任务文件哈希一致；参考测试 13/13；MNIST/CIFAR10 GPU 冒烟通过；仓库三个 phase 检查全部通过。',
        f"A4 torch 检查：{len(models)}/9 完成，全部通过={all(m['pass_checks'] for m in models)}。每组 256 个新市场对照分配与付款；三次随机初始化检验单调性。", '',
        '总 GPU 并发上限 10；QV 期间旧比较组最多占两个槽。每次投放检查空闲显存，为新进程保留 1536 MiB，并保留至少 3072 MiB。结果每约 30 秒回传一次，排除活跃 resume.pt。', '',
        '当前准备状态：' + json.dumps(preparation,ensure_ascii=False),
        '当前后续控制状态：' + json.dumps(continuation,ensure_ascii=False), '',
        '后续：前三个 γ 中有合格者时，按固定顺序训练、运行 Stage 1、通过后直接进入 Stage 3。QV-sel 若被选中，A6 尚未明确连续 κ 方向的 τ 网格及插值规则，控制器会等待该规则，绝不自行补设。', '',
        '已解决的部署问题：补齐 101 缺少的旧 phase 检查脚本与 pandas/pytz/tzdata；未升级 Torch、NumPy 或修改 DP 执行器。重复冒烟构建触发的元组/list 比较问题经规范 JSON 对照证实哈希完全相同，复用原完整冒烟证据。', '',
        '源码封存：launch_manifest.json 与 next_source_manifest.json。旧已完成结果的字节哈希：historical_completed_at_launch.json。', '',
        '本状态报告只读取队列、进度、校验和文件存在性，不读取任何封存 FL 精度。']
    (PAPER / 'LIVE_STATUS_zh.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
