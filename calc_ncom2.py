import argparse
import os
from matplotlib import pyplot as plt
import numpy as np
from sims import get_sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", type=str, help="The simulation case")
    parser.add_argument("--logdir", type=str, default=None,
                        help="The log directory containing the csv files from tensorboard. Default: trainings_ncom/{args.sim}")
    parser.add_argument("--max_steps", type=int, default=None, help="The max number of steps")
    parser.add_argument("--figname", type=str, default="figure.png", help="The figure name")
    args = parser.parse_args()

    sim = get_sim(args.sim, noise=0)
    fdir = os.path.join("trainings_ncom", args.sim) if args.logdir is None else args.logdir
    fpaths = sorted(os.listdir(fdir))
    plt.figure(figsize=(4, 3))
    for ncom, fpath in enumerate(fpaths):
        fp = os.path.join(fdir, fpath)
        data = np.loadtxt(fp, skiprows=1, delimiter=",")[:, 1:]
        diff = data[1:, 0] - data[:-1, 0]
        idx = np.where(diff < 0)[0]
        if len(idx) > 0:
            data = data[:idx[0]]
        if args.max_steps is not None:
            idx0 = np.where(data[:, 0] < args.max_steps)[0][-1]
            data = data[:idx0]
        running_mean = [np.mean(data[i:i + 10, 1]) for i in range(len(data))]
        plt.plot(data[:, 0], data[:, 1], f"C{ncom}", alpha=0.5 / len(fpaths))
        plt.plot(data[:, 0], running_mean, f"C{ncom}", label=f"$n_c = {ncom}$")
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel("L1 value", fontsize=14)
    plt.xlabel("Training step", fontsize=14)
    plt.title(sim.name, fontsize=16)
    plt.gca().set_yscale("log")
    plt.tight_layout()
    plt.savefig(args.figname)

if __name__ == "__main__":
    main()
