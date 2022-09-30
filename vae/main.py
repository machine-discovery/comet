import argparse
import numpy as np
import pytorch_lightning as pl
import torch
from tqdm import tqdm
from methods import get_module
from sims import get_sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, help="The neural network method to be tested")
    parser.add_argument("--sim", type=str, help="The simulation case")
    parser.add_argument("--nstates", type=int, help="The number of latent dimension states")
    parser.add_argument("--ncom", type=int, help="The number of coms")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--max_epochs", type=int, default=20000, help="Maximum number of epochs of training")
    parser.add_argument("--ndset", type=int, default=100, help="Number of simulations to be generated")
    parser.add_argument("--tmax", type=float, default=10.0, help="Max sim time for training")
    parser.add_argument("--dt", type=float, default=0.01, help="The dt to capture the speed")
    parser.add_argument("--version", type=int, default=None, help="The experiment version")
    parser.add_argument("--batch_size", type=int, default=256, help="The max size of a batch")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ndset = args.ndset
    ts = np.linspace(0, args.tmax, int(args.tmax * 10))
    max_epochs = args.max_epochs
    method = args.method
    sim = args.sim
    nstates = args.nstates
    ncom = args.ncom
    dt = args.dt
    batch_size = args.batch_size
    ts_list = [ts for _ in range(ndset)]

    sim_obj = get_sim(sim, dt=args.dt)
    params_list = sim_obj.get_params_list(ndset)
    sim_fcn = sim_obj.simulate

    module = get_module(method, data_shape=sim_obj.data_shape, ncom=ncom, nlatent_dim=nstates, dt=dt)

    # generate the simulations
    states_pp_lst = []
    for i in tqdm(range(ndset)):
        params = [p[i] for p in params_list]
        states_pp = sim_fcn(*params, ts_list[i])  # (ntime_pts, nfeat_tot)
        states_pp_lst.append(states_pp)

    # randomly chosen indices to further shuffle it
    idxs = [78, 12, 77, 29, 58, 17, 70, 45, 30, 13, 56, 90, 67, 84, 99, 25, 1, 62, 8, 28, 85, 79, 48, 18, 74, 91, 3, 6, 65, 47, 61, 24, 34, 87, 80, 31, 73, 63, 50, 53, 59, 89, 7, 96, 43, 0, 55, 52, 11, 71, 72, 66, 46, 4, 23, 26, 2, 32, 35, 20, 95, 88, 57, 92, 75, 22, 83, 64, 76, 97, 54, 14, 9, 16, 86, 41, 27, 68, 44, 33, 81, 60, 21, 93, 36, 69, 42, 15, 37, 39, 98, 49, 82, 5, 10, 40, 94, 19, 38, 51]
    states_pp_lst2 = [states_pp_lst[idx] for idx in idxs]
    states_pp_t = torch.as_tensor(np.concatenate(states_pp_lst2, axis=0), dtype=torch.float32)

    # split into train, val, test
    n = len(states_pp_t)
    train_tensor = states_pp_t[:int(0.7 * n)]
    val_tensor = states_pp_t[int(0.7 * n):int(0.8 * n)]
    test_tensor = states_pp_t[int(0.8 * n):]

    # separate the first (nstates) columns and the last (nstates) columns
    train_dset = torch.utils.data.TensorDataset(train_tensor)
    val_dset = torch.utils.data.TensorDataset(val_tensor)

    # set up the pytorch lightning modules, dataloader, and do the training
    train_dloader = torch.utils.data.DataLoader(train_dset, shuffle=True, batch_size=batch_size)
    val_dloader = torch.utils.data.DataLoader(val_dset, shuffle=False, batch_size=batch_size)
    logger = pl.loggers.tensorboard.TensorBoardLogger("lightning_logs", name="", version=args.version)
    model_chkpt = pl.callbacks.ModelCheckpoint(monitor="val_loss")
    trainer = pl.Trainer(max_epochs=max_epochs, gpus=1, logger=logger, callbacks=[model_chkpt])
    trainer.fit(module, train_dloader, val_dloader)

if __name__ == "__main__":
    main()
