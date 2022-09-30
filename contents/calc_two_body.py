import argparse
import os
from matplotlib import pyplot as plt
import torch
import numpy as np
from methods import load_module
from sims import get_sim


def main():
    torch.manual_seed(5)  # the seed must be different from main.py
    np.random.seed(5)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", type=str, nargs="+", help="The neural network methods to be tested")
    parser.add_argument("--versions", type=int, nargs="+", help="The versions to load the checkpoints")
    parser.add_argument("--figname", type=str, default="figure.png", help="The figure name")
    args = parser.parse_args()

    ndset = 1
    ts = np.linspace(0, 20, 200)
    sim = "two-body"
    methods = args.methods
    versions = args.versions

    # get the simulation information
    sim_obj = get_sim(sim, noise=0.0)
    nstates, ncom = sim_obj.nstates, sim_obj.ncom
    params = [p[0] for p in sim_obj.get_params_list(ndset)]
    sim_fcn = sim_obj.simulate
    states = sim_fcn(*params, ts)[..., :nstates]
    init_state = states[0]

    # calculate the trajectory for every method
    assert len(methods) == len(versions)
    res = {}
    res["True"] = states[:, :nstates // 2]
    for method, version in zip(methods, versions):
        ckpt_path = f"lightning_logs/version_{version}/checkpoints/"
        ckpt_path = os.path.join(ckpt_path, os.listdir(ckpt_path)[0])
        nn = load_module(method, ckpt_path, nstates, ncom)
        states_pred = nn.time_series(init_state, ts)
        res[method.upper()] = states_pred[:, :nstates // 2]

    plt.figure(figsize=(4 * len(res), 3))
    for i, method in enumerate(res.keys()):
        plt.subplot(1, len(res), i + 1)
        st = res[method]
        plt.plot(st[:, 0], st[:, 1], f"C{i}-")
        plt.plot(st[:, 2], st[:, 3], f"C{i}--")
        plt.xlabel("x", fontsize=12)
        plt.ylabel("y", fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.title(method, fontsize=14)
    plt.tight_layout()
    plt.savefig(args.figname)
    plt.close()

if __name__ == "__main__":
    main()
