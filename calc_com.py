import argparse
import os
from matplotlib import pyplot as plt
import torch
import numpy as np
from methods import load_module
from sims import get_sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", type=str, nargs="+", help="The neural network methods to be tested")
    parser.add_argument("--sim", type=str, help="The simulation case")
    parser.add_argument("--versions", type=int, nargs="+", help="The versions to load the checkpoints")
    parser.add_argument("--ncoms", type=int, nargs="+", help="The numbers of coms")
    parser.add_argument("--seed", type=int, default=2, help="Random seed")
    parser.add_argument("--tmax", type=float, default=100.0, help="Max sim time")
    parser.add_argument("--figname", type=str, default="figure.png", help="The figure name")
    args = parser.parse_args()

    torch.manual_seed(args.seed)  # the seed must be different from main.py
    np.random.seed(args.seed)
    
    ndset = 1
    ts = np.linspace(0, args.tmax, int(args.tmax * 10))
    sim = args.sim
    methods = args.methods if args.methods is not None else []
    versions = args.versions if args.versions is not None else []

    # get the simulation information
    sim_obj = get_sim(sim, noise=0)
    ncoms = args.ncoms if args.ncoms is not None else [sim_obj.ncom for i in range(len(versions))]
    nxforce = sim_obj.nxforce
    nstates = sim_obj.nstates
    params_i = [p[0] for p in sim_obj.get_params_list(ndset)]
    params = params_i[:]
    # making the external force a constant
    if nxforce != 0:
        for j in range(nxforce):
            params[-nxforce + j] = lambda ts: params_i[-nxforce + j](ts * 0)
    sim_fcn = sim_obj.simulate
    com_fcns = sim_obj.get_coms()
    states = sim_fcn(*params, ts)[..., :nstates]
    init_state = states[0]

    xforces_lst = []
    if nxforce != 0:
        for j in range(nxforce):
            xforces_lst.append(params[-nxforce + j](ts))
        xforces = np.stack(xforces_lst, axis=-1)

    def get_method_name(method, ncom):
        if method.startswith("comet"):
            if ncom == 0:
                return "NODE"
            elif ncom == sim_obj.ncom:
                return "COMET"
            else:
                return f"COMET ({ncom} coms)"
        else:
            return method.upper()

    # calculate the com for every method
    assert len(ncoms) == len(methods) == len(versions)
    res = {}
    res["True"] = {}
    for com_name, com_fcn in com_fcns.items():
        if nxforce != 0:
            res["True"][com_name] = com_fcn(states, xforces)
        else:
            res["True"][com_name] = com_fcn(states)
    for method, version, ncom in zip(methods, versions, ncoms):
        print(method, version)
        ckpt_path = f"lightning_logs/version_{version}/checkpoints/"
        ckpt_path = os.path.join(ckpt_path, os.listdir(ckpt_path)[0])
        nn = load_module(method, ckpt_path, nstates, ncom, nxforce=nxforce)
        
        if nxforce != 0:
            states_pred = nn.time_series(init_state, ts, params[-1](ts)[:, None])
        else:
            states_pred = nn.time_series(init_state, ts)
        method_name = get_method_name(method, ncom)
        res[method_name] = {}
        for com_name, com_fcn in com_fcns.items():
            if nxforce != 0:
                res[method_name][com_name] = com_fcn(states_pred, xforces)
            else:
                res[method_name][com_name] = com_fcn(states_pred)
                # print(method_name, com_name, states_pred.shape, res[method_name][com_name].shape)

    ncom_fcns = len(com_fcns)
    plt.figure(figsize=(4 * ncom_fcns, 3))
    # ylims = [(-1, 1), (-1, 1), (-1, 1), (0, 2)]
    if sim == "2d-pendulum":
        ylims = [(-1, 1), (0, 2), (-1, 1)]
    elif sim == "two-body":
        ylims = [(-0.5, 0.5), (-0.5, 0.7), (-0.5, 0.5), (0.5, 1.0)]
    else:
        ylims = None
    for i, com_name in enumerate(com_fcns.keys()):
        plt.subplot(1, ncom_fcns, i + 1)
        for method_name in res.keys():
            y = res[method_name][com_name]
            plt.plot(ts[:y.shape[0]], y, label=method_name)
        plt.title(f"{com_name} ({sim})")
        plt.xlabel("Time", fontsize=13)
        if ylims is not None:
            plt.ylim(ylims[i])
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        if i == 0:
            plt.legend()
    plt.tight_layout()
    plt.savefig(args.figname)

if __name__ == "__main__":
    main()
