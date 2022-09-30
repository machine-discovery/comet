import argparse
import os
from matplotlib import pyplot as plt
import torch
import numpy as np
import celluloid as cell
from methods import load_module
from sims import get_sim


def main():
    torch.manual_seed(6)
    np.random.seed(6)

    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", type=str, help="The simulation name")
    parser.add_argument("--ncoms", type=int, nargs="+", help="The number of coms")
    parser.add_argument("--versions", type=int, nargs="+", help="The version to load the checkpoints")
    parser.add_argument("--method", type=str, default="cometcont", help="The NN method")
    parser.add_argument("--figname", type=str, default="figure.png", help="The figure name")
    args = parser.parse_args()

    ndset = 1
    tmax = 20.0
    ts = np.linspace(0, tmax, int(tmax * 10))

    sim_obj = get_sim(args.sim, noise=0.0)
    nstates = sim_obj.nstates
    params = [p[0] for p in sim_obj.get_params_list(ndset)]
    sim_fcn = sim_obj.simulate
    states_dstates = sim_fcn(*params, ts)  # (nt, 2 * nstates)
    states = states_dstates[:, :sim_obj.nstates]
    assert states_dstates.shape[-1] == 2 * sim_obj.nstates
    init_states = states[0]

    all_states = []
    for i, (ncom, version) in enumerate(zip(args.ncoms, args.versions)):
        print(i, ncom, version)
        ckpt_path = f"lightning_logs/version_{version}/checkpoints/"
        ckpt_path = os.path.join(ckpt_path, os.listdir(ckpt_path)[0])
        nn = load_module(args.method, ckpt_path, nstates, ncom)
        states_pred = nn.time_series(init_states, ts)
        all_states.append(states_pred)

    print("All sims done")
    title_fsize = 16
    ticks_fsize = 14
    x = np.linspace(0, 5, len(init_states))
    denorm = lambda states: states * 2 + 1  # denormalization from states to phi, from sims.py
    nimgs = len(args.ncoms) + 1
    ncols = 4
    nrows = int(np.ceil(nimgs / ncols))
    # plot the final value
    fig = plt.figure(figsize=(4 * ncols, 3 * nrows))
    plt.subplot(nrows, ncols, 1)
    plt.plot(x, denorm(states[-1]), color="C0")
    plt.title("True", fontsize=title_fsize)
    plt.ylim([0.95, 4])
    plt.xticks(fontsize=ticks_fsize)
    plt.yticks(fontsize=ticks_fsize)
    for i in range(len(args.ncoms)):
        plt.subplot(nrows, ncols, i + 2)
        plt.plot(x, denorm(all_states[i][-1]), color=f"C{i + 1}")
        plt.title("NODE" if args.ncoms[i] == 0 else f"COMET ({args.ncoms[i]} coms)", fontsize=title_fsize)
        plt.ylim([0.95, 4])
        plt.xticks(fontsize=ticks_fsize)
        plt.yticks(fontsize=ticks_fsize)
    plt.tight_layout()
    plt.savefig(args.figname)
    plt.close()

    # make the animation
    fig = plt.figure(figsize=(4 * ncols, 3 * nrows))
    cam = cell.Camera(fig)
    for it in range(states.shape[0]):
        plt.subplot(nrows, ncols, 1)
        plt.plot(x, denorm(states[it]), color="C0")
        plt.title("True", fontsize=title_fsize)
        plt.ylim([0.95, 4])
        plt.xticks(fontsize=ticks_fsize)
        plt.yticks(fontsize=ticks_fsize)
        for i in range(len(args.ncoms)):
            plt.subplot(nrows, ncols, i + 2)
            plt.plot(x, denorm(all_states[i][it]), color=f"C{i + 1}")
            plt.title("NODE" if args.ncoms[i] == 0 else f"COMET ({args.ncoms[i]} coms)", fontsize=title_fsize)
            plt.ylim([0.95, 4])
            plt.xticks(fontsize=ticks_fsize)
            plt.yticks(fontsize=ticks_fsize)
        plt.tight_layout()
        cam.snap()
    anim = cam.animate()
    anim.save("animation.mp4")

if __name__ == "__main__":
    main()
