import argparse
import os
from matplotlib import pyplot as plt
import torch
import numpy as np
import celluloid as cell
from methods import load_module
from sims import BaseVAESim, get_sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", type=str, help="The simulation case")
    parser.add_argument("--methods", type=str, nargs="+", help="The neural network method to be evaluated")
    parser.add_argument("--versions", type=int, nargs="+", help="The versions to load the checkpoints")
    parser.add_argument("--nstates", type=int, nargs="+", help="The numbers of states")
    parser.add_argument("--ncoms", type=int, nargs="+", help="The numbers of coms")
    parser.add_argument("--names", type=str, nargs="+", help="The methods' names")
    parser.add_argument("--seed", type=int, default=2, help="Random seed")
    parser.add_argument("--tmax", type=float, default=200.0, help="Max sim time")
    parser.add_argument("--figname", type=str, default="figure.png", help="The figure name")
    args = parser.parse_args()

    torch.manual_seed(args.seed)  # the seed must be different from main.py
    np.random.seed(args.seed)

    ndset = 1
    ts = np.linspace(0, args.tmax, 101)
    sim = args.sim

    # get the simulation information
    sim_obj = get_sim(sim)
    assert len(args.methods) == len(args.versions) == len(args.nstates) == len(args.ncoms) == len(args.names)

    # simulate the dynamics
    params_i = [p[0] for p in sim_obj.get_params_list(ndset)]
    params = params_i[:]
    frames = torch.as_tensor(sim_obj.simulate(*params, ts), dtype=torch.float32)  # (nt, 3, nchannels, height, width)
    init_frame = frames[0, :2]  # (2, nchannels, height, width)

    # get the simulated frames
    n = len(args.methods)
    all_frames = {"Ground truth": frames[:, 0]}
    for i in range(n):
        ncom = args.ncoms[i]
        nstates = args.nstates[i]
        method = args.methods[i]
        version = args.versions[i]
        name = args.names[i]

        # get the frames predicted by the methods
        frame_preds = get_frame_pred(init_frame, ts, sim_obj, method, version, ncom, nstates)
        all_frames[name] = frame_preds

    # plot the images
    title_fontsize = 25
    name_fontsize = 25
    nframes = len(all_frames)
    fig = plt.figure(figsize=(3 * nframes, 3))
    cam = cell.Camera(fig)
    for ifig in range(len(ts)):
        for icol, (name, frame_pred) in enumerate(all_frames.items()):
            plt.subplot(1, nframes, icol + 1)
            toshow = sim_obj.to_imshow(frame_pred[ifig])
            plt.imshow(toshow)
            plt.xticks([])
            plt.yticks([])
            if icol == nframes // 2:
                plt.title("t = %.1f" % ts[ifig], fontsize=title_fontsize)
            plt.xlabel(name, fontsize=name_fontsize)
        plt.tight_layout()
        cam.snap()
    anim = cam.animate()
    anim.save("animation.mp4")

def get_frame_pred(init_frame: torch.Tensor, ts: np.ndarray, sim_obj: BaseVAESim, method: str, version: int, ncom: int, nstates: int) -> torch.Tensor:
    # get the frames predicted by the given method
    # init_frames: (2, nchannels, height, width)
    # load the module
    ckpt_path = f"lightning_logs/version_{version}/checkpoints/"
    ckpt_path = os.path.join(ckpt_path, os.listdir(ckpt_path)[0])
    module = load_module(method, ckpt_path=ckpt_path, data_shape=sim_obj.data_shape, ncom=ncom, nlatent_dim=nstates, dt=1e-2)
    module.eval()

    # get the initial zstates
    init_zstate = module.encode(init_frame[None]).squeeze(0)  # (nlatent_dim,)

    # simulate the dynamics and then decode the latent var
    zstates = torch.as_tensor(module.time_series(init_zstate.detach().numpy(), ts), dtype=torch.float32)  # (nt, nlatent_dim)
    frame_pred = module.decode(zstates)  # (nt, nchannels, height, width)
    return frame_pred

if __name__ == "__main__":
    main()
