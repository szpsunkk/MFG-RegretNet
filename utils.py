import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import rc
from cycler import cycler
import copy
from datasets_fl_benchmark import calc_cost_privacy_paper as _cost_privacy_paper

FQ_CONVL = ["FairQuery", "ConvlAggr", "", 1]
ALLIN_CONVL = ["All-in", "ConvlAggr", "", 1]
FQ_OPT = ["FairQuery", "OptAggr", "", 1]
ALLIN_OPT = ["All-in", "OptAggr", "", 1]

REG_CONVL_NSLKDD_IID = ["RegretNet", "ConvlAggr", "model/reg_nslkdd_iid.pt", 1]
MREG_CONVL_NSLKDD_IID = ["M-RegretNet", "ConvlAggr", "model/8-reg_nslkdd_iid.pt", 8]
DM_CONVL_NSLKDD_IID = ["DM-RegretNet", "ConvlAggr", "model/dm-reg_convl_nslkdd_iid.pt", 8]
REG_OPT_NSLKDD_IID = ["RegretNet", "OptAggr", "model/reg_nslkdd_iid.pt", 1]
MREG_OPT_NSLKDD_IID = ["M-RegretNet", "OptAggr", "model/8-reg_nslkdd_iid.pt", 8]
DM_OPT_NSLKDD_IID = ["DM-RegretNet", "OptAggr", "model/dm-reg_opt_nslkdd_iid.pt", 8]


REG_CONVL_NSLKDD_NIID = ["RegretNet", "ConvlAggr", "model/reg_nslkdd_niid.pt", 1]
MREG_CONVL_NSLKDD_NIID = ["M-RegretNet", "ConvlAggr", "model/8-reg_nslkdd_niid.pt", 8]
DM_CONVL_NSLKDD_NIID = ["DM-RegretNet", "ConvlAggr", "model/dm-reg_convl_nslkdd_niid.pt", 8]
REG_OPT_NSLKDD_NIID = ["RegretNet", "OptAggr", "model/reg_nslkdd_niid.pt", 1]
MREG_OPT_NSLKDD_NIID = ["M-RegretNet", "OptAggr", "model/8-reg_nslkdd_niid.pt", 8]
DM_OPT_NSLKDD_NIID = ["DM-RegretNet", "OptAggr", "model/dm-reg_convl_nslkdd_niid.pt", 8]


REG_CONVL_BANK_IID = ["RegretNet", "ConvlAggr", "model/reg_bank_iid.pt", 1]
MREG_CONVL_BANK_IID = ["M-RegretNet", "ConvlAggr", "model/8-reg_bank_iid.pt", 8]
DM_CONVL_BANK_IID = ["DM-RegretNet", "ConvlAggr", "model/dm-reg_convl_bank_iid.pt", 8]
REG_OPT_BANK_IID = ["RegretNet", "OptAggr", "model/reg_bank_iid.pt", 1]
MREG_OPT_BANK_IID = ["M-RegretNet", "OptAggr", "model/8-reg_bank_iid.pt", 8]
DM_OPT_BANK_IID = ["DM-RegretNet", "OptAggr", "model/dm-reg_opt_bank_iid.pt", 8]


REG_CONVL_BANK_NIID = ["RegretNet", "ConvlAggr", "model/reg_bank_niid.pt", 1]
MREG_CONVL_BANK_NIID = ["M-RegretNet", "ConvlAggr", "model/8-reg_bank_niid.pt", 8]
DM_CONVL_BANK_NIID = ["DM-RegretNet", "ConvlAggr", "model/dm-reg_convl_bank_iid.pt", 8]
REG_OPT_BANK_NIID = ["RegretNet", "OptAggr", "model/reg_bank_niid.pt", 1]
MREG_OPT_BANK_NIID = ["M-RegretNet", "OptAggr", "model/8-reg_bank_niid.pt", 8]
DM_OPT_BANK_NIID = ["DM-RegretNet", "OptAggr", "model/dm-reg_opt_bank_iid.pt", 8]


OREG_CONVL_NSLKDD_IID = ["RegretNet", "ConvlAggr",
                         ["model/reg_nslkdd_iid.pt"]
                         + [f"model/regretnet_nslkdd_iid/reg_nslkdd_iid_{i}.pt" for i in range(1, 10)], 1]
TREG_CONVL_NSLKDD_IID = ["2-RegretNet", "ConvlAggr",
                         ["model/2-regretnet_nslkdd_iid/2-reg_nslkdd_iid.pt"]
                        + [f"model/2-regretnet_nslkdd_iid/2-reg_nslkdd_iid_{i}.pt" for i in range(1, 10)], 2]
FREG_CONVL_NSLKDD_IID = ["4-RegretNet", "ConvlAggr",
                         ["model/4-regretnet_nslkdd_iid/4-reg_nslkdd_iid.pt"]
                         + [f"model/4-regretnet_nslkdd_iid/4-reg_nslkdd_iid_{i}.pt" for i in range(1, 10)], 4]
EREG_CONVL_NSLKDD_IID = ["8-RegretNet", "ConvlAggr",
                         ["model/8-reg_nslkdd_iid.pt"]
                         + [f"model/8-regretnet_nslkdd_iid/8-reg_nslkdd_iid_{i}.pt" for i in range(1, 10)], 8]
STREG_CONVL_NSLKDD_IID = ["16-RegretNet", "ConvlAggr",
                          ["model/16-regretnet_nslkdd_iid/16-reg_nslkdd_iid.pt"]
                        + [f"model/16-regretnet_nslkdd_iid/16-reg_nslkdd_iid_{i}.pt" for i in range(1, 10)], 16]


M_EFFECT_NSLKDD_IID = [
    OREG_CONVL_NSLKDD_IID,
    TREG_CONVL_NSLKDD_IID,
    FREG_CONVL_NSLKDD_IID,
    EREG_CONVL_NSLKDD_IID,
    STREG_CONVL_NSLKDD_IID,
]

OREG_CONVL_NSLKDD_NIID = ["RegretNet", "ConvlAggr",
                          ["model/reg_nslkdd_niid.pt"]
                          + [f"model/regretnet_nslkdd_niid/reg_nslkdd_niid_{i}.pt" for i in range(1, 10)], 1]
TREG_CONVL_NSLKDD_NIID = ["2-RegretNet", "ConvlAggr",
                          ["model/2-regretnet_nslkdd_niid/2-reg_nslkdd_niid.pt"]
                          + [f"model/2-regretnet_nslkdd_niid/2-reg_nslkdd_niid_{i}.pt" for i in range(1, 10)], 2]
FREG_CONVL_NSLKDD_NIID = ["4-RegretNet", "ConvlAggr",
                          ["model/4-regretnet_nslkdd_niid/4-reg_nslkdd_niid.pt"]
                          + [f"model/4-regretnet_nslkdd_niid/4-reg_nslkdd_niid_{i}.pt" for i in range(1, 10)], 4]
EREG_CONVL_NSLKDD_NIID = ["8-RegretNet", "ConvlAggr",
                          ["model/8-reg_nslkdd_niid.pt"]
                          + [f"model/8-regretnet_nslkdd_niid/8-reg_nslkdd_niid_{i}.pt" for i in range(1, 10)], 8]
STREG_CONVL_NSLKDD_NIID = ["16-RegretNet", "ConvlAggr",
                           ["model/16-regretnet_nslkdd_niid/16-reg_nslkdd_niid.pt"]
                           + [f"model/16-regretnet_nslkdd_niid/16-reg_nslkdd_niid_{i}.pt" for i in range(1, 10)], 16]


M_EFFECT_NSLKDD_NIID = [
    OREG_CONVL_NSLKDD_NIID,
    TREG_CONVL_NSLKDD_NIID,
    FREG_CONVL_NSLKDD_NIID,
    EREG_CONVL_NSLKDD_NIID,
    STREG_CONVL_NSLKDD_NIID,
]


LEGEND_FONT = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 10,
         }

LABEL_FONT = {'family': 'Times New Roman',
         'weight': 'black',
         'size': 18,
         }

TITLE_FONT = {'family': 'Times New Roman',
         'weight': 'black',
         'size': 18,
         }

def calc_cost_privacy_paper(reports):
    """Cost c(v, epsilon) = v * epsilon per agent. reports: (batch, n_agents, n_items+2)."""
    return _cost_privacy_paper(reports)


def calc_agent_util_privacy_paper(payments, reports):
    """Agent utility u_i = p_i - c(v_i, epsilon_i). payments: (batch, n_agents); reports: (batch, n_agents, n_items+2)."""
    cost = calc_cost_privacy_paper(reports)
    return payments - cost


def generate_critical_budget(reports):
    device = reports.device
    n_items = reports.shape[2] - 2
    n_agents = reports.shape[1]
    n_batches = reports.shape[0]

    vals = reports[:, :, :-2]
    pbudgets = reports[:, :, -2]
    sizes = reports[:, :, -1]
    plosses = pbudgets.view(-1, n_agents, 1).repeat(1, 1, n_items)
    deno = torch.arange(n_items, 0, -1).view(1, 1, n_items).repeat(n_batches, n_agents, 1).to(device)
    plosses = plosses / deno

    unit_prices = vals / plosses
    critical_unit_price = unit_prices.view(n_batches, -1).max(dim=1)[0].view(n_batches, 1)

    critical_budget = (pbudgets * critical_unit_price * sizes).sum(dim=1, keepdims=True)

    return critical_budget

def generate_max_cost(reports):
    return (reports[:, :, -3] * reports[:, :, -1]).sum(dim=1).view(-1, 1)

def calc_deter_violation(allocs):
    n_items = allocs.shape[2]
    n_agents = allocs.shape[1]
    device= allocs.device

    full_allocs = calc_full_allocs(allocs)
    uni_allocs = (torch.ones(full_allocs.shape) * (1/(n_items+1))).to(device)
    dist = torch.linalg.norm(full_allocs - uni_allocs, dim=2) ** 2
    max_dist = ((1.0 - 1 / (n_items + 1)) ** 2 + (0.0 - 1 / (n_items + 1)) ** 2 * n_items)
    return max_dist - dist


def create_combined_misreports(misreports, true_reports):
    n_agents = misreports.shape[1]
    n_items = misreports.shape[2] - 2

    mask = torch.zeros(
        (misreports.shape[0], n_agents, n_agents, n_items+2), device=misreports.device
    )
    for i in range(n_agents):
        mask[:, i, i, :] = 1.0

    tiled_mis = misreports.view(-1, 1, n_agents, n_items+2).repeat(1, n_agents, 1, 1)
    tiled_true = true_reports.view(-1, 1, n_agents, n_items+2).repeat(1, n_agents, 1, 1)

    return mask * tiled_mis + (1.0 - mask) * tiled_true

def allocs_to_plosses(allocs, pbudgets):

    n_items = allocs.shape[2]
    n_agents = allocs.shape[1]
    device = allocs.device
    frac = torch.arange(1, n_items+1, dtype=pbudgets.dtype, device=device) / n_items
    items = pbudgets.view(-1, n_agents, 1).repeat(1, 1, n_items)
    items = items * frac.view(-1, 1, n_items)
    plosses = torch.sum(allocs * items, dim=2)
    # 约束：确保分配的隐私预算不超过声称的预算
    plosses = torch.clamp(plosses, max=pbudgets)
    return plosses


def allocs_instantiate_plosses(allocs, pbudgets):
    n_items = allocs.shape[2]
    n_agents = allocs.shape[1]
    n_batches = allocs.shape[0]
    device = allocs.device

    probs_of_zero_loss = torch.ones(n_batches, n_agents, 1, dtype=allocs.dtype, device=device) - allocs.sum(dim=2).view(n_batches, n_agents, 1)
    full_allocs = torch.cat([probs_of_zero_loss, allocs], dim=2)
    full_allocs = full_allocs.clamp_min_(min=0.0)
    if torch.isnan(full_allocs).sum() > 0.0:
        print("full_allocs contains nan")
        print(full_allocs)

    if torch.isnan(full_allocs).sum() > 0.0:
        print("full_allocs contains inf")
        print(full_allocs)

    results = torch.multinomial(full_allocs.view(-1, n_items + 1), 1).view(-1, n_agents)
    per_pbudgets = pbudgets.view(-1, n_agents) / n_items
    plosses = per_pbudgets * results

    return plosses, results


def calc_full_allocs(allocs):
    n_agents = allocs.shape[1]
    n_batches = allocs.shape[0]
    n_items = allocs.shape[2]
    device = allocs.device

    probs_of_zero_loss = torch.ones(n_batches, n_agents, 1, dtype=allocs.dtype, device=device) - allocs.sum(dim=2).view(n_batches, n_agents, 1)
    full_allocs = torch.cat([probs_of_zero_loss, allocs], dim=2)
    full_allocs = full_allocs.clamp_min_(min=0.0)
    if torch.isnan(full_allocs).sum() > 0.0:
        print("full_allocs contains nan")
        print(full_allocs)

    if torch.isnan(full_allocs).sum() > 0.0:
        print("full_allocs contains inf")
        print(full_allocs)

    return full_allocs.view(-1, n_agents, n_items+1)


def calc_agent_util(reports, agent_allocations, payments, instantiation=False, cost_from_plosses=False, true_valuation_for_cost=None):
    """
    cost_from_plosses: if True and n_items==1, use privacy-paper cost c(v,eps)=v*eps with eps=allocated plosses.
    true_valuation_for_cost: optional (batch, n_agents) or (batch, n_agents, 1); when cost_from_plosses and set, use for v in cost (e.g. true v in misreport context).
    """
    n_batches = reports.shape[0]
    n_agents = reports.shape[1]
    n_items = reports.shape[2] - 2
    device = reports.device

    valuations = reports[:, :, :-2]
    pbudgets = reports[:, :, -2].view(-1, n_agents)
    sizes = reports[:, :, -1].view(-1, n_agents)

    if not instantiation:
        privacy_losses = allocs_to_plosses(agent_allocations, pbudgets)
        if cost_from_plosses and n_items == 1:
            v_for_cost = true_valuation_for_cost if true_valuation_for_cost is not None else valuations[:, :, 0]
            if v_for_cost.dim() == 3:
                v_for_cost = v_for_cost.squeeze(-1)
            costs = v_for_cost * privacy_losses
        else:
            costs = torch.sum(agent_allocations * valuations, dim=2) * sizes
    else:
        privacy_losses, ins_results = allocs_instantiate_plosses(agent_allocations, pbudgets)
        if cost_from_plosses and n_items == 1:
            v_for_cost = true_valuation_for_cost if true_valuation_for_cost is not None else valuations[:, :, 0]
            if v_for_cost.dim() == 3:
                v_for_cost = v_for_cost.squeeze(-1)
            costs = v_for_cost * privacy_losses
        else:
            zero_vals = torch.zeros(n_agents * n_batches).view(-1, 1).to(device)
            valuations_flat = valuations.view(-1, n_items)
            full_vals = torch.cat((zero_vals, valuations_flat), dim=1)
            ins_results = ins_results.view(-1)
            idxs = torch.arange(ins_results.shape[0])
            costs = full_vals[idxs, ins_results].view(-1, n_agents) * sizes
    util = payments - costs
    violations = (pbudgets - privacy_losses) < 0
    util = torch.where(violations, torch.zeros(violations.shape, device=violations.device), util)
    return util


def make_monotonic(vals, forward=True):
    if forward:
        for i in range(1, vals.shape[1]):
            vals[:, i] = torch.clamp(vals[:, i], min=vals[:, i-1])
    else:
        for i in range(0, vals.shape[1]-1):
            vals[:, i] = torch.clamp(vals[:, i], max=vals[:, i+1])
    return vals

def optimize_misreports(
    model, current_reports, current_misreports, budget, val_type, misreport_iter=10, lr=1e-1, train=True, instantiation=False, cost_from_plosses=False
):
    current_misreports.requires_grad_(True)
    true_pbudgets = current_reports[:, :, -2]
    true_sizes = current_reports[:, :, -1]

    for i in range(misreport_iter):
        model.zero_grad()
        agent_utils = tiled_misreport_util(current_misreports, current_reports, model, budget, val_type, instantiation=instantiation, cost_from_plosses=cost_from_plosses)

        (misreports_grad,) = torch.autograd.grad(agent_utils.sum(), current_misreports)
        misreports_grad = torch.where(torch.isnan(misreports_grad), torch.full_like(misreports_grad, 0), misreports_grad)

        with torch.no_grad():
            current_misreports += lr * misreports_grad
            if train:
                model.module.clamp_op(current_misreports)
            else:
                model.clamp_op(current_misreports)
            fake_vals = current_misreports[:, :, :-2]
            current_misreports[:, :, :-2] = make_monotonic(fake_vals)

            fake_pbudgets = current_misreports[:, :, -2]
            fake_pbudgets[fake_pbudgets > true_pbudgets] = true_pbudgets[fake_pbudgets > true_pbudgets]

            fake_sizes = current_misreports[:, :, -1]
            fake_sizes[fake_sizes > true_sizes] = true_sizes[fake_sizes > true_sizes]
            fake_sizes[fake_sizes < 1] = 1

    return current_misreports

def create_real_reports(misreports, val_type):
    n_batches = misreports.shape[0]
    n_agents = misreports.shape[1]
    n_items = misreports.shape[2] - 2
    device = misreports.device
    # val_type = val_type.view(n_batches, n_agents, 1)
    # real_reports = torch.cat([misreports, val_type], dim=2)
    pbudgets = misreports[:, :, -2]
    real_vals = pbudgets.view(n_batches, n_agents, 1).repeat(1, 1, n_items)
    frac = torch.arange(1, n_items + 1, device=device).reshape(1, 1, n_items) / n_items
    real_vals = real_vals * frac

    real_vals = real_vals.view(n_batches * n_agents, n_items)
    val_type = val_type.view(n_batches * n_agents, 2)
    real_vals[val_type[:, 0] == 0] = real_vals[val_type[:, 0] == 0] ** 2
    real_vals[val_type[:, 0] == 1] = 2 * real_vals[val_type[:, 0] == 1] ** 0.5
    real_vals[val_type[:, 0] == 2] = 2 * real_vals[val_type[:, 0] == 2]
    real_vals[val_type[:, 0] == 3] = torch.exp(real_vals[val_type[:, 0] == 3]) - 1
    real_vals = (real_vals * val_type[:, 1].view(-1, 1)).view(n_batches, n_agents, n_items)

    real_reports = torch.cat((real_vals, misreports[:, :, -2:]), dim=2)
    # real_reports[:, :, :-2] = real_vals

    return real_reports

def tiled_misreport_util(current_misreports, current_reports, model, budget, val_type, instantiation=False, cost_from_plosses=False):
    n_agents = current_reports.shape[1]
    n_items = current_reports.shape[2] - 2
    batch_size = current_reports.shape[0]

    agent_idx = list(range(n_agents))
    tiled_misreports = create_combined_misreports(
        current_misreports, current_reports
    )
    flatbatch_tiled_misreports = tiled_misreports.view(-1, n_agents, n_items+2)
    tiled_budget = budget.view(-1, 1).repeat(1, n_agents).view(-1, 1)
    allocations, payments = model((flatbatch_tiled_misreports, tiled_budget))
    reshaped_payments = payments.view(
        -1, n_agents, n_agents
    )
    reshaped_allocations = allocations.view(-1, n_agents, n_agents, n_items)
    agent_payments = reshaped_payments[:, agent_idx, agent_idx]
    agent_payments = agent_payments.repeat_interleave(n_agents, dim=0)
    # For each (batch, deviating_agent) we need the full (n_agents, n_items) allocation so batch dim = batch_size * n_agents to match real_reports
    agent_allocations = reshaped_allocations.view(batch_size * n_agents, n_agents, n_items)
    real_reports = create_real_reports(current_misreports, val_type)
    # Match batch dim to agent_allocations (batch_size * n_agents, n_agents, ...)
    real_reports = real_reports.repeat_interleave(n_agents, dim=0)
    true_valuation_for_cost = None
    if cost_from_plosses and n_items == 1:
        true_valuation_for_cost = current_reports[:, :, 0].repeat_interleave(n_agents, dim=0)
    agent_utils = calc_agent_util(
        real_reports, agent_allocations, agent_payments, instantiation=instantiation,
        cost_from_plosses=cost_from_plosses, true_valuation_for_cost=true_valuation_for_cost
    )
    # agent_utils is (batch_size * n_agents, n_agents); for train_loop we need (batch_size, n_agents): utility of deviating agent per (batch, agent)
    # row i = b*10+a gives utilities when agent a deviated; we need agent_utils[i, a] -> index (i, i % n_agents)
    i = torch.arange(batch_size * n_agents, device=agent_utils.device)
    agent_utils = agent_utils[i, i % n_agents].view(batch_size, n_agents)
    return agent_utils


def plot(x_list, y_list, labels, title, file_name, xlabel, ylabel, yscale='linear', xscale="linear"):
    if len(labels) == 4 and ylabel == 'model accuracy':
        market_color_ls = [
            ('s', 'r', 'solid', 0, 'r'),
            ('h', 'g', 'solid', 1, 'g'),
            ('d', 'b', 'solid', 2, 'b'),
            ('o', 'k', 'solid', 3, 'k'),
        ]
    elif len(labels) == 4:
        market_color_ls = [
            ('s', 'r', 'solid', 0, 'r'),
            ('h', 'g', 'solid', 1, 'g'),
            ('o', 'c', 'solid', 2, 'c'),
            ('d', 'b', '--', 3, "none"),
        ]
    elif len(labels) == 3:
        market_color_ls = [
            ('o', 'g'),
            ('d', 'b', '--'),
            ('s', 'r'),
        ]
    else:
        market_color_ls = [
            ('v', 'b', 'solid', 0, 'b'),
            ('h', 'g', 'solid', 1, 'g'),
            ('d', 'm', 'solid', 2, 'm'),
            ('^', 'r', 'solid', 3, 'r'),
            ('o', 'c', 'solid', 4, 'c'),
            ('s', 'k', 'solid', 5, 'k'),
            ('*', 'y', 'solid', 6, 'y'),
            ('X', 'olive', 'solid', 7, 'olive')
        ]

    if xscale == "log":
        fig, ax = plt.subplots()
        ax.set_xscale('log', base=2)
        ax.set_yscale(yscale)
        for i in range(len(x_list)):
            ax.plot(x_list[i], y_list[i], label=labels[i], marker=market_color_ls[i][0], color=market_color_ls[i][1], alpha=1.0)
    else:
        for i in range(len(x_list)):
            plt.plot(x_list[i], y_list[i], label=labels[i], marker=market_color_ls[i][0], color=market_color_ls[i][1],
                     linestyle=market_color_ls[i][2], zorder=market_color_ls[i][3], markerfacecolor=market_color_ls[i][4])


    plt.yscale(yscale)
    plt.tick_params(labelsize=8)
    plt.legend(prop=LEGEND_FONT)
    plt.xlabel(xlabel, LABEL_FONT)
    plt.ylabel(ylabel, LABEL_FONT)
    plt.title(title, TITLE_FONT)
    plt.rcParams['savefig.dpi'] = 400
    plt.rcParams['figure.dpi'] = 400
    plt.savefig(file_name, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_budget_acc(x_list, y_list, labels, title, file_name, yscale='logit'):
    plot(x_list, y_list, labels, title, file_name, 'financial budget factor', 'model accuracy', yscale=yscale)

def plot_rnd_acc(x_list, y_list, labels, title, file_name, yscale='logit'):
    plot(x_list, y_list, labels, title, file_name, 'training round', 'model accuracy', yscale=yscale)

def plot_budget_mse(x_list, y_list, labels, title, file_name, yscale='log'):
    plot(x_list, y_list, labels, title, file_name, 'financial budget factor', 'finite error bound', yscale=yscale)


def plot_budget_invalid_rate(x_list, y_list, labels, title, file_name, yscale='linear'):
    plot(x_list, y_list, labels, title, file_name, 'financial budget factor', 'invalid gradient rate', yscale=yscale)


def plot_n_agents_mse(x_list, y_list, labels, title, file_name, yscale='log'):
    plot(x_list, y_list, labels, title, file_name, 'number of data owners', 'finite error bound', yscale=yscale)

def plot_m_guarantees(x_list, y_list, labels, title, file_name, yscale='linear', xscale="log"):
    plot(x_list, y_list, labels, title, file_name, 'parameter M', 'violation degree', yscale=yscale, xscale=xscale)
    

def plot_bar(x_list, y_list, labels, title, file_name, xlabel, ylabel):
    pattern = ('//', '\\')

    tick_ls = []
    for n_items in x_list:
        tick_ls.append(str(n_items))

    cols = list(range(len(x_list)))

    for i in range(len(y_list)):
        plt.bar(cols, y_list[i], width=0.4, label=labels[i], tick_label=tick_ls, hatch=pattern[i])
        for j in range(len(cols)):
            cols[j] = cols[j] + 0.4

    plt.tick_params(labelsize=8)
    plt.legend(prop=LEGEND_FONT)
    plt.xlabel(xlabel, LABEL_FONT)
    plt.ylabel(ylabel, LABEL_FONT)
    plt.title(title, TITLE_FONT)
    plt.rcParams['savefig.dpi'] = 400
    plt.rcParams['figure.dpi'] = 400
    plt.savefig(file_name, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_n_items_mse(x_list, y_list, labels, title, file_name):
    plot_bar(x_list, y_list, labels, title, file_name, xlabel='M-Unit', ylabel='finite error bound')


