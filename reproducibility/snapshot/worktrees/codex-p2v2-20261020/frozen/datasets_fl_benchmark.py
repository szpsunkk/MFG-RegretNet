"""
FL benchmark datasets for privacy paper experiments (Phase 1).
- MNIST, CIFAR-10: Dirichlet(alpha) non-IID split.
- Shakespeare: optional LEAF-style loader for next-character prediction.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, Subset


# ---------------------------------------------------------------------------
# MNIST
# ---------------------------------------------------------------------------

def load_mnist(root="data/mnist", download=True):
    """Load MNIST train and test. Returns (train_dataset, test_dataset)."""
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST(
        root=root, train=True, download=download, transform=transform
    )
    test_set = datasets.MNIST(
        root=root, train=False, download=download, transform=transform
    )
    # Ensure .targets exists (PyTorch 1.x uses .targets)
    if not hasattr(train_set, 'targets'):
        train_set.targets = train_set.train_labels
    if not hasattr(test_set, 'targets'):
        test_set.targets = test_set.test_labels
    return train_set, test_set


def load_fmnist(root="data/fmnist", download=True):
    """Load Fashion-MNIST train and test."""
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    train_set = datasets.FashionMNIST(
        root=root, train=True, download=download, transform=transform
    )
    test_set = datasets.FashionMNIST(
        root=root, train=False, download=download, transform=transform
    )
    if not hasattr(train_set, 'targets'):
        train_set.targets = train_set.train_labels
    if not hasattr(test_set, 'targets'):
        test_set.targets = test_set.test_labels
    return train_set, test_set


# ---------------------------------------------------------------------------
# CIFAR-10
# ---------------------------------------------------------------------------

def load_cifar10(root="data/cifar10", download=True):
    """Load CIFAR-10 train and test. Returns (train_dataset, test_dataset)."""
    from torchvision import datasets, transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ),
    ])
    train_set = datasets.CIFAR10(
        root=root, train=True, download=download, transform=transform_train
    )
    test_set = datasets.CIFAR10(
        root=root, train=False, download=download, transform=transform_test
    )
    if not hasattr(train_set, 'targets'):
        train_set.targets = train_set.train_labels
    if not hasattr(test_set, 'targets'):
        test_set.targets = test_set.test_labels
    return train_set, test_set


# ---------------------------------------------------------------------------
# Dirichlet non-IID split (alpha=0.5 for privacy paper)
# ---------------------------------------------------------------------------

def dirichlet_split(
    dataset,
    n_clients,
    n_classes,
    alpha=0.5,
    min_size=10,
    seed=None
):
    """
    Split dataset among n_clients using Dirichlet(alpha) over class proportions.
    Returns: idxs_client_dict = { client_id: list of sample indices }.
    """
    if seed is not None:
        np.random.seed(seed)
    data_size = len(dataset)
    idxs = np.arange(data_size)
    labels = np.array(dataset.targets)

    # sort by label
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :].astype(int)
    labels = idxs_labels[1, :]
    n_per_class = np.array([np.sum(labels == c) for c in range(n_classes)])

    # per-class indices
    idxs_by_class = []
    start = 0
    for c in range(n_classes):
        end = start + n_per_class[c]
        idxs_by_class.append(idxs[start:end].tolist())
        start = end

    # Dirichlet proportions: (n_classes, n_clients)
    proportions = np.random.dirichlet(np.repeat(alpha, n_clients), size=n_classes)
    # ensure each client gets at least min_size samples
    n_per_class_client = (proportions * (n_per_class.reshape(-1, 1) - min_size * n_clients)).astype(np.int64)
    n_per_class_client = np.maximum(n_per_class_client, 0)
    # give remainder to last client so totals match
    for c in range(n_classes):
        remainder = len(idxs_by_class[c]) - n_per_class_client[c].sum()
        if remainder > 0:
            n_per_class_client[c, -1] += remainder
        elif remainder < 0:
            n_per_class_client[c, -1] = max(0, n_per_class_client[c, -1] + remainder)

    idxs_client_dict = {i: [] for i in range(n_clients)}
    for c in range(n_classes):
        class_idxs = idxs_by_class[c]
        for i in range(n_clients):
            take = n_per_class_client[c, i]
            if take <= 0:
                continue
            if i == n_clients - 1 and len(class_idxs) > 0:
                chosen = class_idxs
            else:
                take = min(take, len(class_idxs))
                if take == 0:
                    continue
                chosen = np.random.choice(class_idxs, size=take, replace=False).tolist()
                class_idxs = list(set(class_idxs) - set(chosen))
            idxs_client_dict[i].extend(chosen)

    return idxs_client_dict


def get_client_subsets(dataset, idxs_client_dict):
    """Return list of Subset(dataset, indices) per client."""
    return [Subset(dataset, idxs_client_dict[i]) for i in sorted(idxs_client_dict.keys())]


def get_client_data_loaders(dataset, idxs_client_dict, batch_size=None):
    """Return list of DataLoader per client (for local training)."""
    from torch.utils.data import DataLoader
    subsets = get_client_subsets(dataset, idxs_client_dict)
    if batch_size is None:
        batch_size = 64
    return [
        DataLoader(sub, batch_size=min(batch_size, len(sub)), shuffle=True)
        for sub in subsets
    ]


# ---------------------------------------------------------------------------
# Shakespeare (next-character prediction)
# LEAF format: all_data.json with user_data[user_id]['x'], ['y'] (sequence, next-char)
# ---------------------------------------------------------------------------

class ShakespeareDataset(Dataset):
    """Dataset of (sequence, next_char_id) for next-character prediction."""

    def __init__(self, sequences, next_chars, seq_len=80):
        # sequences: list of (seq_len+1) int arrays (char indices)
        self.seq_len = seq_len
        self.x = []  # (seq_len,) each
        self.y = []  # scalar next char
        for seq in sequences:
            seq = np.array(seq, dtype=np.int64)
            for i in range(len(seq) - seq_len):
                self.x.append(seq[i : i + seq_len])
                self.y.append(seq[i + seq_len])
        self.x = torch.tensor(np.array(self.x)) if self.x else torch.zeros(0, seq_len, dtype=torch.long)
        self.y = torch.tensor(self.y, dtype=torch.long) if self.y else torch.zeros(0, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    @property
    def targets(self):
        """For compatibility with dirichlet_split (we split by 'user' not class)."""
        return self.y.numpy()


def load_shakespeare_leaf(data_dir="data/shakespeare", seq_len=80):
    """
    Load Shakespeare from LEAF-style all_data.json.
    Expects: data_dir/all_data.json with keys user_data[user]['x'] (list of lists), ['y'].
    Returns (train_dataset_full, test_dataset_full, vocab_size) or (None, None, 0) if missing.
    """
    import json
    path = os.path.join(data_dir, "all_data.json")
    if not os.path.isfile(path):
        return None, None, 0
    with open(path, "r") as f:
        data = json.load(f)
    user_data = data.get("user_data", {})
    if not user_data:
        return None, None, 0
    all_x, all_y = [], []
    vocab = set()
    for user, ud in user_data.items():
        xs = ud.get("x", [])
        ys = ud.get("y", [])
        for x, y in zip(xs, ys):
            seq = x if isinstance(x, list) else list(x)
            if len(seq) > seq_len:
                all_x.append(seq[: seq_len + 1])
                all_y.append(y if isinstance(y, int) else y[0])
                vocab.update(seq)
                vocab.add(y if isinstance(y, int) else y[0])
    if not all_x:
        return None, None, 0
    vocab_size = max(vocab) + 1
    # Build one big dataset (we can split by user later for non-IID)
    full_seqs = [s for s in all_x]
    full_next = [all_y[i] for i in range(len(all_x))]
    dataset = ShakespeareDataset(full_seqs, full_next, seq_len=seq_len)
    # No train/test split in LEAF per file; use 90/10
    n = len(dataset)
    perm = np.random.permutation(n)
    train_n = int(0.9 * n)
    train_set = Subset(dataset, perm[:train_n])
    test_set = Subset(dataset, perm[train_n:])
    train_set.targets = dataset.y[perm[:train_n]].numpy()
    test_set.targets = dataset.y[perm[train_n:]].numpy()
    return train_set, test_set, vocab_size


def load_shakespeare_dummy(seq_len=80, num_samples=1000, vocab_size=80):
    """Dummy Shakespeare-like data for testing (random sequences)."""
    np.random.seed(42)
    # Each sequence must have length seq_len+1 so ShakespeareDataset yields one (x,y) per seq
    sequences = [
        np.random.randint(0, vocab_size, size=seq_len + 1).tolist()
        for _ in range(num_samples)
    ]
    next_chars = [s[-1] for s in sequences]
    dataset = ShakespeareDataset(sequences, next_chars, seq_len=seq_len)
    n = len(dataset)
    perm = np.random.permutation(n)
    train_n = int(0.9 * n)
    train_set = Subset(dataset, perm[:train_n])
    test_set = Subset(dataset, perm[train_n:])
    train_set.targets = dataset.y[perm[:train_n]].numpy()
    test_set.targets = dataset.y[perm[train_n:]].numpy()
    return train_set, test_set, vocab_size


# ---------------------------------------------------------------------------
# Privacy paper bid generation (v ~ U[0,1], epsilon ~ U[0.1, 5], c = v * epsilon)
# ---------------------------------------------------------------------------

def generate_privacy_paper_bids(
    n_agents,
    n_items,
    num_profiles,
    v_min=0.0,
    v_max=1.0,
    eps_min=0.1,
    eps_max=5.0,
    size_min=10,
    size_max=500,
    seed=None
):
    """
    Generate bid profiles for privacy paper setting.
    Paper: v_i ~ Uniform[0,1], epsilon_i ~ Uniform[0.1, 5], c(v, epsilon) = v * epsilon.
    Returns: tensor of shape (num_profiles, n_agents, n_items + 2)
    - reports[:, i, :n_items] = valuation per item (we use single item = v_i for all items for simplicity)
    - reports[:, i, n_items]   = epsilon_i (pbudget)
    - reports[:, i, n_items+1] = data size (for aggregation weight)
    So for n_items=1: (batch, n_agents, 3) with [v, epsilon, size].
    """
    if seed is not None:
        np.random.seed(seed)
    reports = np.zeros((num_profiles, n_agents, n_items + 2))
    for b in range(num_profiles):
        for i in range(n_agents):
            v = np.random.uniform(v_min, v_max)
            eps = np.random.uniform(eps_min, eps_max)
            size = np.random.randint(size_min, size_max + 1) if size_max >= size_min else 100
            # one "item" = privacy level; valuation for that item = v
            for k in range(n_items):
                reports[b, i, k] = v
            reports[b, i, n_items] = eps
            reports[b, i, n_items + 1] = size
    return torch.tensor(reports, dtype=torch.float32)


def calc_cost_privacy_paper(reports):
    """
    Cost c(v, epsilon) = v * epsilon per agent.
    reports: (batch, n_agents, n_items+2); pbudget at [..., -2], valuation we use first item [..., 0].
    Returns: (batch, n_agents) cost per agent.
    """
    v = reports[:, :, 0]
    eps = reports[:, :, -2]
    return v * eps
