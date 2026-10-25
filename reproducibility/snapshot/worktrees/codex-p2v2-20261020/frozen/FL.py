import math
import torch
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import random
import torch.utils.data as Data
import copy
import matplotlib.pyplot as plt
import numpy as np


def laplace_noise_like(tensor, scale):
    """
    Sample noise ~ Laplace(0, scale) with the same shape as tensor.
    Use this instead of Laplace(loc=grad, scale=...).sample(), which validates
    a huge `loc` tensor and can hang on CNN-scale parameters.
    """
    if isinstance(scale, torch.Tensor):
        scale = float(scale.detach().float().cpu().item())
    u = torch.rand_like(tensor) - 0.5
    # Inverse-CDF sample for Laplace(0, b), u in (-1/2, 1/2)
    noise = -scale * torch.sign(u) * torch.log(
        1.0 - 2.0 * u.abs().clamp(max=0.5 - 1e-12) + 1e-30
    )
    return noise


class Arguments():
    def __init__(self):
        self.local_batch_size = 10
        self.test_batch_size = 1000
        self.rounds = 100
        self.lr = 0.01
        self.no_cuda = False
        self.seed = 0
        self.log_interval = 50
        self.save_model = False
        self.submit_grad = True
        self.L = 1.0
        self.sensi = 2.0 * self.L
        # RQ4 / Fed: multiple local passes stabilize test accuracy vs rounds
        self.local_epochs = 1
        self.local_batch_size = 64
        self.use_cuda = not self.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # Logits for nn.CrossEntropyLoss (do not use log_softmax here)
        return x

class Logistic(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Logistic, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out = self.linear(x)
        out = self.sig(out)
        return out


class CIFAR10Net(nn.Module):
    """Simple CNN for CIFAR-10 (32x32x3 -> 10 classes)."""

    def __init__(self):
        super(CIFAR10Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ShakespeareLSTM(nn.Module):
    """LSTM for next-character prediction (privacy paper Shakespeare)."""

    def __init__(self, vocab_size=80, embed_size=64, hidden_size=128, num_layers=2, dropout=0.2):
        super(ShakespeareLSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        # x: (batch, seq_len) long
        emb = self.embed(x)
        out, _ = self.lstm(emb)
        logits = self.fc(out[:, -1, :])
        return F.log_softmax(logits, dim=1)


def ldp_fed_sgd(model, args, plosses, weights, local_sets, rnd):
    #     torch.cuda.empty_cache()
    updates = []
    keys = list(model.state_dict().keys())
    device = plosses.device
    total_num_samples = 0
    plosses = plosses.view(-1)
    n_agents = plosses.shape[0]
    weights = weights.view(-1)
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    # print(plosses)

    global_model = copy.deepcopy(model).to(device)
    state = global_model.state_dict()

    # step 1: selection

    if (plosses == 0.0).all():
        return model

    for i in range(n_agents):
        # step 2: broadcasting

        if plosses[i] > 0.0:
            local_model = copy.deepcopy(model).to(device)

            # step 3: local training
            local_model.train()
            local_set = local_sets[i]

            X = local_set[0].to(device)
            Y = local_set[1].to(device).long()
            if X.dtype in (torch.float32, torch.float64):
                X = X.float()
            else:
                X = X.long()

            epsi = plosses[i]

            optimizer = optim.SGD(local_model.parameters(), lr=args.lr)
            optimizer.zero_grad()
            pred_Y = local_model(X)
            criter = nn.CrossEntropyLoss()
            loss = criter(pred_Y, Y)
            loss.backward()
            # Do not call optimizer.step() before reading grads (would clear .grad).

            # step 4: submission — noisy gradient (Laplace) to global state
            nn.utils.clip_grad_norm_(local_model.parameters(), args.L, 1.0)

            for param in local_model.named_parameters():
                g = param[1].grad
                if g is None:
                    continue
                noise = laplace_noise_like(g, args.sensi / epsi)
                noised_grad = (g + noise).to(device)
                state[param[0]] = state[param[0]] - noised_grad * weights[i] * args.lr


    global_model.load_state_dict(state)

    return global_model


def pag_fl_alg2_round(model, args, plosses, local_sets, delta=0.01, eps_min=0.1):
    """
    Paper Algorithm 2 (PAG-FL deploy): from global w, each participant runs local SGD;
    then Gaussian noise on full local weights ew_i = w_i + N(0, σ_i^2 I) with
    σ_i = sqrt(2 ln(1.25/δ)) · Δ_f / max(ε_out_i, ε_min); aggregate
    w_G = Σ_i α_i ew_i with α_i = ε_out_i / Σ_j ε_out_j over participants.
    `plosses[i]` is the realized ε_out from the auction (same role as in ldp_fed_sgd).
    """
    device = next(model.parameters()).device
    plosses = plosses.view(-1).float().to(device)
    n_agents = plosses.shape[0]
    participants = [i for i in range(n_agents) if plosses[i] > 0]
    if not participants:
        return model, 0.0

    # δ must be in (0, 1.25) so ln(1.25/δ) ≥ 0; clamp careless args
    delta_safe = min(max(float(delta), 1e-9), 1.24)
    coef = math.sqrt(2.0 * math.log(1.25 / delta_safe))
    noised_states = []
    eps_used = []
    losses = []
    # Single floor for α and σ (avoids inf/nan when eps_min=0 or tiny plosses)
    eps_floor = max(float(eps_min), 1e-8)

    for i in participants:
        lm = copy.deepcopy(model).to(device)
        lm.train()
        X = local_sets[i][0].to(device)
        Y = local_sets[i][1].to(device).long()
        if X.dtype in (torch.float32, torch.float64):
            X = X.float()

        eff_eps = max(float(plosses[i].item()), eps_floor)
        eps_used.append(eff_eps)

        le = max(1, int(getattr(args, "local_epochs", 1)))
        lbs = max(1, int(getattr(args, "local_batch_size", 64)))
        n_samples = X.size(0)
        bs = max(1, min(lbs, n_samples))
        loader = Data.DataLoader(
            Data.TensorDataset(X, Y),
            batch_size=bs,
            shuffle=True,
            drop_last=False,
        )
        optimizer = optim.SGD(lm.parameters(), lr=args.lr)
        round_losses = []
        for _ep in range(le):
            for xb, yb in loader:
                optimizer.zero_grad()
                pred_Y = lm(xb)
                loss = nn.CrossEntropyLoss()(pred_Y, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(lm.parameters(), args.L, 1.0)
                optimizer.step()
                round_losses.append(loss.item())
        losses.extend(round_losses)

        sigma = coef * float(args.L) / eff_eps
        sd = lm.state_dict()
        new_sd = {}
        for k, v in sd.items():
            if v.is_floating_point():
                new_sd[k] = v + torch.randn_like(v, device=v.device, dtype=v.dtype) * sigma
            else:
                new_sd[k] = v.clone()
        noised_states.append(new_sd)

    s_eps = sum(eps_used)
    alpha = [e / s_eps for e in eps_used]
    merged = {}
    keys = noised_states[0].keys()
    for k in keys:
        v0 = noised_states[0][k]
        if v0.is_floating_point():
            merged[k] = sum(
                alpha[j] * noised_states[j][k] for j in range(len(participants))
            )
        else:
            # Long buffers etc.: weighted sum would promote to float — keep one copy
            merged[k] = noised_states[0][k].clone()

    out = copy.deepcopy(model).to(device)
    out.load_state_dict(merged)
    return out, float(np.mean(losses)) if losses else 0.0


def test(model, test_set, args, rnd):
    model.eval()
    device = args.device
    test_loss = 0.0
    correct = 0
    # print(len(test_set))
    data_loader = Data.DataLoader(dataset=test_set, batch_size=10000)
    num_samples = 0
    with torch.no_grad():
        for i, test_data in enumerate(data_loader):
            x = test_data[0].to(device)
            y = test_data[1].to(device).long()
            if x.dtype in (torch.float32, torch.float64):
                x = x.float()
            else:
                x = x.long()
            # x = test_data[0].to(device).view(-1, 784)
            # y = test_data[1].to(device)
            num_samples += len(x)
            pred_y = model(x)
            # test_loss += F.nll_loss(pred_y, y.long(), reduction='sum')
            criter = nn.CrossEntropyLoss()
            test_loss += criter(pred_y, y.long())
            pred = pred_y.argmax(1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()

        test_loss /= num_samples
        # print('\n Round: {}  Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #     rnd, test_loss, correct, num_samples,
        #     100. * correct / num_samples))
    return correct / num_samples

