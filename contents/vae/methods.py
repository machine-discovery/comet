from abc import abstractmethod
from typing import Callable, List, Optional, Tuple, Type
import numpy as np
import scipy.integrate
import torch
import functorch
import pytorch_lightning as pl


class MLP2(torch.nn.Module):
    def __init__(self, ninp: int, nout: int, nhidden: int = 256):
        super().__init__()
        self._act = torch.nn.LogSigmoid()
        self._lin1 = torch.nn.Linear(ninp, nhidden, bias=False)
        self._lin2 = torch.nn.Linear(nhidden, nhidden, bias=False)
        self._lin3 = torch.nn.Linear(nhidden, nhidden, bias=False)
        self._lin4 = torch.nn.Linear(nhidden, nout, bias=False)
        for mod in [self._lin1, self._lin2, self._lin3, self._lin4]:
            torch.nn.init.orthogonal_(mod.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._act(self._lin1(x))
        x = x + self._act(self._lin2(x))
        x = x + self._act(self._lin3(x))
        x = self._lin4(x)
        return x

class Reshape(torch.nn.Module):
    def __init__(self, data_shape: Tuple[int, ...], nlast_dim: int):
        super().__init__()
        self._data_shape = data_shape
        self._nlast_dim = nlast_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(*x.shape[:-self._nlast_dim], *self._data_shape)
        return x

class BaseMethod(torch.nn.Module):
    @abstractmethod
    def calc_loss(self, zstates: torch.Tensor, dzstates_calc: torch.Tensor) -> torch.Tensor:
        # calculate the loss
        pass

    @abstractmethod
    def forward(self, zstates: torch.Tensor) -> torch.Tensor:
        # calculate dzstates
        pass

class CoMet(BaseMethod):
    def __init__(self, nstates: int, ncom: int, nhidden: int = 250):
        super().__init__()
        self._nn = MLP2(nstates, nstates + ncom, nhidden=nhidden)
        self._nstates = nstates
        self._ncom = ncom
        self._loss = torch.nn.MSELoss()

    def _get_dzs(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # returns the dynamics (the ortho dzstates and the original zstates)
        states = states.requires_grad_()
        def _get_com_dstates(states):
            # states: (nstates,), xforce: (nxforce,)
            nnout = self._nn(states)
            dstates = nnout[..., :self._nstates]  # (nstates,)
            com = nnout[..., self._nstates:]  # (ncom,)
            return com, (dstates, com)
        if self._ncom == 0:
            dstates = self._nn(states)
            return dstates, dstates, None
        else:
            jac_fcn = functorch.jacrev(_get_com_dstates, 0, has_aux=True)
            for _ in range(states.ndim - 1):
                jac_fcn = functorch.vmap(jac_fcn)
            dcom_jac, (dstates, com) = jac_fcn(states)  # dcom_jac: (..., ncom, nstates), dstates: (..., nstates)
            dcom_jac = dcom_jac.transpose(-2, -1).contiguous()  # (..., nstates, ncom)
            dcom_jac_dstates = torch.cat((dcom_jac, dstates[..., None]), dim=-1)  # (..., nstates, ncom + 1)
            q, r = torch.linalg.qr(dcom_jac_dstates)  # q: (..., nstates, ncom + 1), r: (..., ncom + 1, ncom + 1)
            dstates_ortho = q[..., -1] * r[..., -1, -1:]  # (..., nstates)
            return dstates_ortho, dstates, dcom_jac

    def forward(self, zstates: torch.Tensor) -> torch.Tensor:
        return self._get_dzs(zstates)[0]
    
    def calc_loss(self, zstates: torch.Tensor, dzstates_calc: torch.Tensor) -> torch.Tensor:
        dzstates_ortho, dzstates, dcom_jac = self._get_dzs(zstates)

        if dcom_jac is None:
            comet_reg = 0
        else:
            comet_reg = torch.mean(torch.einsum("...sc,...s->...c", dcom_jac, dzstates) ** 2)

        # comet losses
        comet_loss0 = torch.mean(self._loss(dzstates_ortho, dzstates_calc))
        comet_loss1 = torch.mean(self._loss(dzstates, dzstates_calc))
        return (comet_loss0 + comet_loss1 + comet_reg) * 0.5  # the contribution from comet_reg usually not as significant as the other 2

class HNN(BaseMethod):
    def __init__(self, nstates: int, nhidden: int = 250):
        super().__init__()
        assert nstates % 2 == 0
        self._nn = MLP2(nstates, 1, nhidden=nhidden)
        self._s2 = nstates // 2
        self._loss = torch.nn.MSELoss()

    def forward(self, zstates: torch.Tensor) -> torch.Tensor:
        # zstates: (..., nstates)
        fcn = functorch.grad(lambda zstates: self._nn(zstates).sum())
        for _ in range(zstates.ndim - 1):
            fcn = functorch.vmap(fcn)
        dhamilt = fcn(zstates)
        # first half is for dq and the second half is for dp
        dstates = torch.cat((dhamilt[..., self._s2:], -dhamilt[..., :self._s2]), dim=-1)
        return dstates

    def calc_loss(self, zstates: torch.Tensor, dzstates_calc: torch.Tensor) -> torch.Tensor:
        # zstates, dzstates_calc: (..., nstates)
        dzstates = self.forward(zstates)
        hamilt_loss = self._loss(dzstates, dzstates_calc)
        cc_loss = self._loss(zstates[..., self._s2:], dzstates_calc[..., :self._s2])  # p == dq
        return hamilt_loss + cc_loss

class NSF(BaseMethod):
    def __init__(self, nstates: int):
        super().__init__()
        assert nstates % 2 == 0
        self._nn = MLP2(nstates, 1)
        self._ynn = MLP2(nstates, nstates)
        self._loss = torch.nn.MSELoss()

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        # returns the dynamics

        # get the inverse Y matrix
        fcn = functorch.jacrev(self._ynn)
        for _ in range(states.ndim - 1):
            fcn = functorch.vmap(fcn)
        yjac = fcn(states)  # (..., nstates, nstates)
        w_mT = torch.linalg.inv(yjac - yjac.transpose(-2, -1))  # (..., nstates, nstates)

        # get the hamiltonian
        hfcn = functorch.grad(lambda states: self._nn(states).sum())
        for _ in range(states.ndim - 1):
            hfcn = functorch.vmap(hfcn)
        dhamilt = hfcn(states)

        # calculate the dstates
        dstates = torch.einsum("...ab,...b->...a", w_mT, dhamilt)
        return dstates

    def calc_loss(self, zstates: torch.Tensor, dzstates_calc: torch.Tensor) -> torch.Tensor:
        # zstates, dzstates_calc: (..., nstates)
        dzstates = self.forward(zstates)
        hamilt_loss = self._loss(dzstates, dzstates_calc)
        return hamilt_loss

class VAE(pl.LightningModule):
    # data_shape: (nchannels, height, width)
    def __init__(self, data_shape: Tuple[int, int, int], method: BaseMethod, nlatent_dim: int, dt: float, nhidden: int = 250):
        super().__init__()
        self._dshape = (2 * data_shape[0], data_shape[1], data_shape[2])  # the channels are concatted for subsequent frames
        self._enc = torch.nn.Sequential(Reshape((-1,), 3), MLP2(np.prod(self._dshape), nlatent_dim))
        self._dec = torch.nn.Sequential(MLP2(nlatent_dim, np.prod(self._dshape)), Reshape(self._dshape, 1))
        self._method = method
        self._dt = dt
        self._nlatent_dim = nlatent_dim
        self._loss = torch.nn.MSELoss()
        self.register_buffer("_max_dframe", torch.tensor([-1.]).max())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4)

    def preprocess(self, frames: torch.Tensor) -> torch.Tensor:
        assert frames.size(-4) == 2

        frame0 = frames[..., 0, :, :, :]
        frame1 = frames[..., 1, :, :, :]
        dframe = (frame1 - frame0) / self._dt  # (..., nchannels, height, width)
        if torch.all(self._max_dframe < 0):
            self._max_dframe = dframe.abs().max().detach()
        dframe = dframe / self._max_dframe
        fdframe = torch.cat((frame0, dframe), dim=-3)  # (..., 2 * nchannels, height, width)
        return fdframe

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        # obtain the latent variable from 2 subsequent frames
        # frames: (..., 2, nchannels, height, width)
        # returns: (..., nlatent_dim)

        fdframe = self.preprocess(frames)  # (..., 2 * nchannels, height, width)
        # # frames2: (..., 2 * nchannels, height, width)
        # frames2 = frames.reshape(*frames.shape[:-4], *self._dshape)
        zstates = self._enc(fdframe)
        return zstates
    
    def decode(self, zstates: torch.Tensor) -> torch.Tensor:
        # obtain a frame from the latent state (it will lose the velocity information)
        dec_frames = self._dec(zstates)  # (..., 2 * nchannels, height, width)
        frames = dec_frames.reshape(*dec_frames.shape[:-3], 2, -1, *dec_frames.shape[-2:])  # (..., 2, nchannels, height, width)
        frame = frames[..., 0, :, :, :]
        return frame

    def calc_loss(self, batch: Tuple[torch.Tensor]):
        # returns the dynamics and the com
        # imgs: (batch_size, 3, nchannels, height, width)
        imgs, = batch

        # frames: (batch_size, 2, nchannels, height, width)
        frames = imgs[..., :2, :, :, :]
        next_frames = imgs[..., 1:, :, :, :]
        # zstates: (batch_size, nlatent_dim)
        zstates = self.encode(frames)
        next_zstates = self.encode(next_frames)

        # calculate the dzstates and the comet loss
        dzstates_calc = (next_zstates - zstates) / self._dt
        method_loss = self._method.calc_loss(zstates, dzstates_calc)

        # ae loss
        zs = zstates + torch.randn_like(zstates) * 1e-1  # (..., nlatent_dim)
        frames_pred = self._dec(zs)
        ae_loss = torch.mean(self._loss(frames_pred, self.preprocess(frames)))
        # frames_pred = self.decode(zs)
        # ae_loss0 = torch.mean(self._loss(frames_pred, frames[:, 0]))
        return method_loss, ae_loss

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        # returns the dynamics dzstates
        return self._method.forward(states)

    def time_series(self, init_zstate: np.ndarray, ts: np.ndarray) -> np.ndarray:
        # init_state: (nstates,)
        # ts: (ntime_pts,)
        # returns: (ntime_pts, nstates)
        def dynamics(t: float, zstate: np.ndarray) -> np.ndarray:
            zstate_t = torch.as_tensor(zstate, dtype=torch.float32)[None, :]
            dzstates = self.forward(zstate_t).squeeze(0)
            # dzstates = nnout[..., :self._nlatent_dim].squeeze(0)
            return dzstates.detach().numpy()
        ret = scipy.integrate.solve_ivp(dynamics, [ts[0], ts[-1]], init_zstate, t_eval=ts, rtol=1e-9, atol=1e-9)
        zstates = ret.y.T
        return zstates

    def training_step(self, train_batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        method_loss, ae_loss = self.calc_loss(train_batch)
        self.log("training_method_loss", method_loss)
        self.log("training_ae_loss", ae_loss)
        return method_loss + 10 * ae_loss

    def validation_step(self, val_batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        method_loss, ae_loss = self.calc_loss(val_batch)
        val_loss = method_loss + 10 * ae_loss
        self.log("val_loss", val_loss)
        self.log("val_method_loss", method_loss)
        self.log("val_ae_loss", ae_loss)
        return val_loss

def get_module(method: str, data_shape: Tuple[int, int, int], ncom: int, nlatent_dim: int, dt: float) -> pl.LightningModule:
    if method == "comet":
        method_obj = CoMet(nlatent_dim, ncom)
        module = VAE(data_shape=data_shape, method=method_obj, nlatent_dim=nlatent_dim, dt=dt)
    elif method == "hnn":
        method_obj = HNN(nlatent_dim)
        module = VAE(data_shape=data_shape, method=method_obj, nlatent_dim=nlatent_dim, dt=dt)
    elif method == "nsf":
        method_obj = NSF(nlatent_dim)
        module = VAE(data_shape=data_shape, method=method_obj, nlatent_dim=nlatent_dim, dt=dt)
    else:
        raise RuntimeError(f"Unknown method: {method}")
    return module

def load_module(method: str, ckpt_path: str, data_shape: Tuple[int, int, int], ncom: int, nlatent_dim: int, dt: float) -> pl.LightningModule:
    if method == "comet":
        method_obj = CoMet(nlatent_dim, ncom)
        nn = VAE.load_from_checkpoint(ckpt_path, method=method_obj, data_shape=data_shape, ncom=ncom, nlatent_dim=nlatent_dim, dt=dt)
    elif method == "hnn":
        method_obj = HNN(nlatent_dim)
        nn = VAE.load_from_checkpoint(ckpt_path, method=method_obj, data_shape=data_shape, ncom=ncom, nlatent_dim=nlatent_dim, dt=dt)
    elif method == "nsf":
        method_obj = NSF(nlatent_dim)
        nn = VAE.load_from_checkpoint(ckpt_path, method=method_obj, data_shape=data_shape, ncom=ncom, nlatent_dim=nlatent_dim, dt=dt)
    else:
        raise RuntimeError(f"Unknown method: {method}")
    return nn
