import argparse
import os
from matplotlib import pyplot as plt
import torch
import numpy as np
from methods import load_module
from sims import get_sim


def main():
    torch.manual_seed(3)  # the seed must be different from main.py
    np.random.seed(3)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", type=str, help="The simulation case")
    parser.add_argument("--versions", type=int, nargs="+", help="The versions to load the checkpoints")
    parser.add_argument("--figname", type=str, default="figure.png", help="The figure name")
    args = parser.parse_args()

    ndset = 100
    ts = np.linspace(0, 100, 100)
    sim = args.sim
    versions = args.versions

    # get the simulation information
    sim_obj = get_sim(sim, noise=0)
    nstates, ncom = sim_obj.nstates, sim_obj.ncom
    sim_fcn = sim_obj.simulate
    states_dstates_lst = []
    for i in range(ndset):
        params = [p[i] for p in sim_obj.get_params_list(ndset)]
        states_dstates_lst.append(torch.as_tensor(sim_fcn(*params, ts), dtype=torch.float32))
    states_dstates = torch.cat(states_dstates_lst, dim=0)
    states, dstates = torch.split(states_dstates, nstates, dim=-1)

    losses_lst = []
    for ncom, version in enumerate(versions):
        print(version)
        ckpt_path = f"lightning_logs/version_{version}/checkpoints/"
        ckpts = os.listdir(ckpt_path)
        losses_seed = []
        for ckpt in ckpts:
            ckpt_full_path = os.path.join(ckpt_path, ckpt)
            if ncom == 0:
                nn = load_module("node", ckpt_full_path, nstates, ncom)
                dstates_pred = nn.forward(states)  # (..., nstates)
            else:
                nn = load_module("comet", ckpt_full_path, nstates, ncom)
                dstates_pred = nn.forward(states)[0]  # (..., nstates)
            loss = float(torch.mean((dstates - dstates_pred) ** 2).detach().numpy())  # (,)
            losses_seed.append(loss)  # [nseed]
        losses_lst.append(losses_seed)  # [nversions, nseed]

    min_nseed = min([len(ls) for ls in losses_lst])
    losses_lst2 = [ls[:min_nseed] for ls in losses_lst]
    losses = np.stack(losses_lst2, axis=0)  # (nversions, nseed)
    rel_losses = losses / losses[0]  # (nversions, nseed)
    log_mean_rel_losses = np.mean(np.log(rel_losses), axis=-1)  # (nversions)
    log_std_rel_losses = np.std(np.log(rel_losses), axis=-1)  # (nversions)
    mean_rel_losses = np.exp(log_mean_rel_losses)
    ncoms = list(range(len(versions)))
    plt.figure(figsize=(4, 3))
    # plt.plot(ncoms, rel_losses, marker="o")
    plt.errorbar(ncoms, mean_rel_losses, (mean_rel_losses - np.exp(log_mean_rel_losses - log_std_rel_losses), np.exp(log_mean_rel_losses + log_std_rel_losses) - mean_rel_losses), marker="o")
    # plt.errorbar(ncoms, log_mean_rel_losses, log_std_rel_losses, marker="o")
    plt.xlabel("Number of CoMs", fontsize=14)
    plt.ylabel("Relative mean values of L1", fontsize=14)
    plt.title(sim_obj.name, fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.gca().set_yscale("log")
    plt.tight_layout()
    plt.savefig(args.figname)
    plt.close()

if __name__ == "__main__":
    main()
