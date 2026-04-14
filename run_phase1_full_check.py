#!/usr/bin/env python3
"""
Phase 1 完整过程检查：按实验规划逐项验证数据、划分、bid、成本与 FL 管线。
"""
import os
import sys

def run(check_name, fn):
    try:
        fn()
        print(f"  [PASS] {check_name}")
        return True
    except Exception as e:
        print(f"  [FAIL] {check_name}: {e}")
        return False

def main():
    ok = 0
    total = 0

    # -------------------------------------------------------------------------
    # 1. 隐私论文 bid 生成与形状
    # -------------------------------------------------------------------------
    total += 1
    def _bids():
        from datasets_fl_benchmark import generate_privacy_paper_bids
        b = generate_privacy_paper_bids(5, 1, 4, seed=42)
        assert b.shape == (4, 5, 3), b.shape
        assert b[:, :, 0].min() >= 0 and b[:, :, 0].max() <= 1
        assert b[:, :, 1].min() >= 0.1 and b[:, :, 1].max() <= 5
    if run("1. 隐私论文 bid 生成 (shape, v∈[0,1], ε∈[0.1,5])", _bids):
        ok += 1

    # -------------------------------------------------------------------------
    # 2. 成本 c = v*ε
    # -------------------------------------------------------------------------
    total += 1
    def _cost():
        from datasets_fl_benchmark import generate_privacy_paper_bids, calc_cost_privacy_paper
        b = generate_privacy_paper_bids(5, 1, 4, seed=42)
        c = calc_cost_privacy_paper(b)
        for i in range(4):
            for j in range(5):
                assert abs(c[i,j].item() - b[i,j,0].item()*b[i,j,1].item()) < 1e-5
    if run("2. 成本 c(v,ε)=v*ε", _cost):
        ok += 1

    # -------------------------------------------------------------------------
    # 3. 效用 u = p - c (utils)
    # -------------------------------------------------------------------------
    total += 1
    def _util():
        from datasets_fl_benchmark import generate_privacy_paper_bids
        from utils import calc_agent_util_privacy_paper
        b = generate_privacy_paper_bids(3, 1, 2, seed=1)
        p = b[:, :, 0] * 0.5  # dummy payments
        u = calc_agent_util_privacy_paper(p, b)
        assert u.shape == (2, 3)
    if run("3. 效用 calc_agent_util_privacy_paper(payments, reports)", _util):
        ok += 1

    # -------------------------------------------------------------------------
    # 4. Shakespeare dummy 非空且形状正确
    # -------------------------------------------------------------------------
    total += 1
    def _shakespeare_dummy():
        import torch
        from datasets_fl_benchmark import load_shakespeare_dummy
        train, test, vocab = load_shakespeare_dummy(seq_len=80, num_samples=100, vocab_size=80)
        assert len(train) > 0 and len(test) > 0
        x, y = train[0]
        assert x.shape == (80,) and (x.dtype == torch.long or x.dtype == torch.int64)
    if run("4. Shakespeare dummy 非空、seq 形状 (80,)", _shakespeare_dummy):
        ok += 1

    # -------------------------------------------------------------------------
    # 5. dirichlet_split (datasets_fl_benchmark) 与 get_client_subsets
    # -------------------------------------------------------------------------
    total += 1
    def _dirichlet():
        from datasets_fl_benchmark import dirichlet_split, get_client_subsets
        import numpy as np
        class DummyDataset:
            def __init__(self, n, n_classes):
                self.targets = np.random.randint(0, n_classes, size=n)
            def __len__(self):
                return len(self.targets)
        ds = DummyDataset(1000, 10)
        idxs = dirichlet_split(ds, 5, 10, alpha=0.5, seed=42)
        subs = get_client_subsets(ds, idxs)
        assert len(subs) == 5
        total_idx = sum(len(idxs[i]) for i in range(5))
        assert total_idx == 1000
    if run("5. dirichlet_split + get_client_subsets (5 clients, 10 classes)", _dirichlet):
        ok += 1

    # -------------------------------------------------------------------------
    # 6. generate_clients("Shakespeare", non-IID)
    # -------------------------------------------------------------------------
    total += 1
    def _gen_shakespeare():
        from client import Clients
        os.makedirs("data/shakespeare_dummy", exist_ok=True)
        c = Clients()
        c.dirs = "data/shakespeare_dummy/"
        c.filename = "phase1_check.json"
        d = c.generate_clients("Shakespeare", 1, 5, iid=False, alpha=0.5)
        assert len(d) == 5
        for i in range(5):
            assert "data_indices" in d[i] and "privacy_budget" in d[i]
    if run("6. generate_clients(Shakespeare, non-IID, 5 clients)", _gen_shakespeare):
        ok += 1

    # -------------------------------------------------------------------------
    # 7. generate_clients("MNIST", non-IID) — 需 torchvision
    # -------------------------------------------------------------------------
    try:
        from torchvision import datasets
        HAS_TORCHVISION = True
    except ImportError:
        HAS_TORCHVISION = False

    if HAS_TORCHVISION:
        total += 1
        def _gen_mnist():
            from client import Clients
            os.makedirs("data/mnist/niid", exist_ok=True)
            c = Clients()
            c.dirs = "data/mnist/niid/"
            c.filename = "phase1_check_mnist.json"
            c.min_pbudget = 0.1
            c.max_pbudget = 5.0
            d = c.generate_clients("MNIST", 1, 10, iid=False, alpha=0.5)
            assert len(d) == 10
        if run("7. generate_clients(MNIST, non-IID, 10 clients)", _gen_mnist):
            ok += 1
    else:
        print("  [SKIP] 7. generate_clients(MNIST) — 需要 torchvision")

    # -------------------------------------------------------------------------
    # 8. MNIST 单轮 FL (extr_noniid_dirt + Net + ldp_fed_sgd + test)
    # -------------------------------------------------------------------------
    if HAS_TORCHVISION:
        total += 1
        def _mnist_fl():
            import torch
            from datasets_fl_benchmark import load_mnist, get_client_subsets
            from client import extr_noniid_dirt
            from FL import Net, Arguments, ldp_fed_sgd, test
            from torch.utils.data import DataLoader
            torch.manual_seed(42)
            train_set, test_set = load_mnist()
            idxs = extr_noniid_dirt(train_set, 10, 10, alpha=0.5)
            subsets = get_client_subsets(train_set, idxs)
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            args = Arguments()
            args.device = dev
            args.rounds = 1
            args.lr = 0.01
            args.L = 1.0
            args.sensi = 2.0 * args.L
            local_sets = []
            for sub in subsets:
                n = len(sub)
                if n == 0:
                    raise RuntimeError("Empty client")
                dl = DataLoader(sub, batch_size=max(1, min(64, n)), shuffle=True)
                x, y = next(iter(dl))
                local_sets.append((x, y))
            model = Net().to(dev)
            plosses = torch.ones(10, device=dev) * 0.5
            weights = torch.ones(10, device=dev) / 10
            model = ldp_fed_sgd(model, args, plosses, weights, [local_sets], 0)
            acc = test(model, test_set, args, 0)
            assert 0 <= acc <= 1
        if run("8. MNIST 单轮 FL (Dirichlet + Net + ldp_fed_sgd + test)", _mnist_fl):
            ok += 1
    else:
        print("  [SKIP] 8. MNIST 单轮 FL — 需要 torchvision")

    # -------------------------------------------------------------------------
    # 9. CIFAR-10 单轮 FL
    # -------------------------------------------------------------------------
    if HAS_TORCHVISION:
        total += 1
        def _cifar_fl():
            import torch
            from datasets_fl_benchmark import load_cifar10, get_client_subsets
            from client import extr_noniid_dirt
            from FL import CIFAR10Net, Arguments, ldp_fed_sgd, test
            from torch.utils.data import DataLoader
            torch.manual_seed(42)
            train_set, test_set = load_cifar10()
            idxs = extr_noniid_dirt(train_set, 10, 10, alpha=0.5)
            subsets = get_client_subsets(train_set, idxs)
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            args = Arguments()
            args.device = dev
            args.rounds = 1
            args.lr = 0.01
            local_sets = []
            for sub in subsets:
                n = len(sub)
                if n == 0:
                    raise RuntimeError("Empty client")
                dl = DataLoader(sub, batch_size=max(1, min(64, n)), shuffle=True)
                x, y = next(iter(dl))
                local_sets.append((x, y))
            model = CIFAR10Net().to(dev)
            plosses = torch.ones(10, device=dev) * 0.5
            weights = torch.ones(10, device=dev) / 10
            model = ldp_fed_sgd(model, args, plosses, weights, [local_sets], 0)
            acc = test(model, test_set, args, 0)
            assert 0 <= acc <= 1
        if run("9. CIFAR-10 单轮 FL (Dirichlet + CIFAR10Net + ldp_fed_sgd + test)", _cifar_fl):
            ok += 1
    else:
        print("  [SKIP] 9. CIFAR-10 单轮 FL — 需要 torchvision")

    # -------------------------------------------------------------------------
    # 10. FL.py 中 LSTM 输入为 long（Shakespeare 兼容）
    # -------------------------------------------------------------------------
    total += 1
    def _lstm_dtype():
        from FL import ShakespeareLSTM
        import torch
        m = ShakespeareLSTM(vocab_size=80, embed_size=8, hidden_size=16, num_layers=1, dropout=0)
        x = torch.randint(0, 80, (2, 80))
        out = m(x)
        assert out.shape == (2, 80)
    if run("10. ShakespeareLSTM 前向 (long 输入, log_softmax 输出)", _lstm_dtype):
        ok += 1

    # -------------------------------------------------------------------------
    print()
    print(f"Phase 1 过程检查: {ok}/{total} 通过")
    if ok < total and not HAS_TORCHVISION:
        print("未安装 torchvision 时，项 7/8/9 会跳过；安装后重新运行可做完整检查。")
    return 0 if ok == total else 1

if __name__ == "__main__":
    sys.exit(main())
