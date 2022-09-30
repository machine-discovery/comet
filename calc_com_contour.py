import argparse
import os
from matplotlib import pyplot as plt
import torch
import numpy as np
from methods import load_module
from sims import get_sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", type=str, help="The simulation case")
    parser.add_argument("--version", type=int, help="The version to load the checkpoints")
    args = parser.parse_args()
    
    # load the comet
    nstates, ncom = 2, 1
    ckpt_path = f"lightning_logs/version_{args.version}/checkpoints/"
    ckpt_path = os.path.join(ckpt_path, os.listdir(ckpt_path)[0])
    nn = load_module("comet", ckpt_path, nstates, ncom)

    # get the bounds to be simulated
    ns = 100
    sim = get_sim(args.sim, noise=0)
    params_names = sim.get_states_name()
    assert sim.ncom == ncom and sim.nstates == nstates, "This script can only work with simulation with 2 states and 1 com"
    params_list = sim.get_params_list(10000)
    ub = np.array([np.max(p) for p in params_list])  # (nstates,)
    lb = np.array([np.min(p) for p in params_list])

    # get the coms in the states within the bounds
    states_lst = [np.linspace(lb[i], ub[i], ns) for i in range(nstates)]
    states = np.stack(np.meshgrid(*states_lst, indexing="ij"), axis=-1)  # (ns0, ns1, nstates)
    states = np.reshape(states, (ns * ns, nstates))  # (ns0 * ns1, nstates)
    coms = nn.forward(torch.as_tensor(states, dtype=torch.float32), xforce=None)[2]  # (ns0 * ns1, coms)
    true_com = np.reshape(list(sim.get_coms().values())[0](states), (ns, ns))
    assert coms.shape[-1] == ncom
    coms_np = coms[..., 0].reshape(ns, ns).detach().numpy()  # (ns0, ns1)

    # get the levels to produce the contour
    if args.sim == "lotka-volterra":
        lstates = np.stack((np.linspace(1.1, 2.0, 10), np.linspace(1.1, 2.0, 10)), axis=-1)
        levels = np.sort(nn.forward(torch.as_tensor(lstates, dtype=torch.float32), xforce=None)[2][..., 0].detach().numpy())
        true_levels = np.sort(list(sim.get_coms().values())[0](lstates))
    elif args.sim == "mass-spring":
        lstates = np.stack((np.linspace(0.05, 0.5, 10), np.linspace(0.05, 0.5, 10)), axis=-1)
        levels = np.sort(nn.forward(torch.as_tensor(lstates, dtype=torch.float32), xforce=None)[2][..., 0].detach().numpy())
        true_levels = np.sort(list(sim.get_coms().values())[0](lstates))
    else:
        levels = 10
        true_levels = 10

    # plot the coms
    plt.figure(figsize=(7, 3))
    plt.subplot(1, 2, 1)
    plt.contour(states_lst[0], states_lst[1], coms_np.T, levels=levels)
    plt.title(f"{sim.name} (COMET)", fontsize=16)
    plt.ylabel(params_names[1], fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(params_names[0], fontsize=14)
    plt.xticks(fontsize=14)
    plt.subplot(1, 2, 2)
    plt.contour(states_lst[0], states_lst[1], true_com.T, levels=true_levels)
    plt.title(f"{sim.name} (True)", fontsize=16)
    # plt.ylabel(params_names[1], fontsize=14)
    plt.yticks([])
    plt.xlabel(params_names[0], fontsize=14)
    plt.xticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(f"com-contour-{args.sim}")
    plt.close()

if __name__ == "__main__":
    main()
