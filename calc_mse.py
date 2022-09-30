import argparse
import os
from matplotlib import pyplot as plt
import torch
import numpy as np
from methods import load_module
from sims import get_sim


def main():
    torch.manual_seed(1)  # the seed must be different from main.py
    np.random.seed(1)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, help="The neural network method to be tested")
    parser.add_argument("--sim", type=str, help="The simulation case")
    parser.add_argument("--version", type=int, help="The version to load the checkpoints")
    parser.add_argument("--ncom", type=int, default=None, help="Number of coms")
    parser.add_argument("--tmax", type=float, default=100.0, help="Time max")
    parser.add_argument("--ndset", type=int, default=100, help="Number of simulations")
    args = parser.parse_args()

    ndset = args.ndset
    ts = np.linspace(0, args.tmax, int(args.tmax * 10))
    sim = args.sim
    method = args.method
    version = args.version

    sim_obj = get_sim(sim)
    nstates, ncom = sim_obj.nstates, sim_obj.ncom
    params_list = sim_obj.get_params_list(ndset)
    sim_fcn = sim_obj.simulate
    nxforce = sim_obj.nxforce
    ncom = ncom if args.ncom is None else args.ncom

    ckpt_path = f"lightning_logs/version_{version}/checkpoints/"
    ckpt_path = os.path.join(ckpt_path, os.listdir(ckpt_path)[0])
    nn = load_module(method, ckpt_path, nstates, ncom, nxforce=nxforce)

    losses = []
    for i in range(ndset):
        params_i = [p[i] for p in params_list]
        params = params_i[:]
        # making the external force a constant
        if nxforce != 0:
            for j in range(nxforce):
                params[-nxforce + j] = lambda ts: params_i[-nxforce + j](ts * 0)
        states = sim_obj.postproc(sim_fcn(*params, ts)[..., :nstates])
        if nxforce != 0:
            states_pred = sim_obj.postproc(nn.time_series(states[0], ts, params[-1](ts)[:, None]))
        else:
            states_pred = sim_obj.postproc(nn.time_series(states[0], ts))
        if len(states_pred) != len(states):
            loss = np.sqrt(np.mean((states[:len(states_pred)] - states_pred) ** 2))  # rmse
            print(f"{i} failed")
        else:
            loss = np.sqrt(np.mean((states - states_pred) ** 2))  # rmse
        if i % max(1, ndset // 10) == 0:
            print(i, loss)
        losses.append(loss)
    print(f"{sim} ({method}, {version}):", np.median(losses), np.percentile(losses, 97.5) - np.median(losses), np.median(losses) - np.percentile(losses, 2.5))
    np.savetxt(f"{sim}_{method}_{version}.txt", losses)

if __name__ == "__main__":
    main()
