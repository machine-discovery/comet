from typing import Callable, Optional, Tuple, Type, List
import pytorch_lightning as pl
import torch
import numpy as np
import scipy.integrate
import functorch
from sims import get_sim
from functools import partial
from torch.autograd import grad
from torch.autograd.functional import hessian, jacobian


class MLP(torch.nn.Module):
    def __init__(self, ninp: int, nout: int, ndepths: int = 4,
                 nhidden: int = 50,
                 bias: bool = True,
                 activation: Type[torch.nn.Module] = torch.nn.ReLU,
                 ):
        super().__init__()

        # build the sequential model
        layers: List[torch.nn.Module] = []
        for i in range(ndepths):
            ninp_layer = ninp if i == 0 else nhidden
            nout_layer = nhidden if i < ndepths - 1 else nout
            linear_layer = torch.nn.Linear(ninp_layer, nout_layer, bias=bias)
            layers.append(linear_layer)
            if i < ndepths - 1:
                layers.append(activation())

        self._module = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._module.forward(x)

class CoMet(pl.LightningModule):
    def __init__(self, nstates: int, ncom: int, nxforce: int = 0, nhidden: int = 250):
        super().__init__()
        assert nstates > ncom
        self._nstates = nstates
        self._nxforce = nxforce
        self._nn = MLP(nstates + nxforce, nstates + ncom, activation=torch.nn.LogSigmoid, nhidden=nhidden)
        self._loss = torch.nn.MSELoss()
        self._ncom = ncom

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4)

    def forward(self, states: torch.Tensor, xforce: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # returns the dynamics and the com
        states = states.requires_grad_()
        def _get_com_dstates(states, xforce):
            # states: (nstates,), xforce: (nxforce,)
            nnout = self._nn(torch.cat((states, xforce), dim=-1))
            dstates = nnout[..., :self._nstates]  # (nstates,)
            com = nnout[..., self._nstates:]  # (ncom,)
            return com, (dstates, com)
        if xforce is None:
            xforce = torch.zeros((*states.shape[:-1], 0), dtype=states.dtype, device=states.device)
        if self._ncom == 0:
            dstates = self._nn(torch.cat((states, xforce), dim=-1))
            return dstates, dstates, None, None
        else:
            jac_fcn = functorch.vmap(functorch.jacrev(_get_com_dstates, 0, has_aux=True))
            dcom_jac, (dstates, com) = jac_fcn(states, xforce)  # dcom_jac: (..., ncom, nstates), dstates: (..., nstates)
            dcom_jac = dcom_jac.transpose(-2, -1).contiguous()  # (..., nstates, ncom)
            dcom_jac_dstates = torch.cat((dcom_jac, dstates[..., None]), dim=-1)  # (..., nstates, ncom + 1)
            q, r = torch.linalg.qr(dcom_jac_dstates)  # q: (..., nstates, ncom + 1), r: (..., ncom + 1, ncom + 1)
            dstates_ortho = q[..., -1] * r[..., -1, -1:]  # (..., nstates)
            return dstates_ortho, dstates, com, dcom_jac

    def time_series(self, init_state: np.ndarray, ts: np.ndarray, xforce: Optional[np.ndarray] = None,
                    state_postproc: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> np.ndarray:
        # init_state: (nstates,)
        # ts: (ntime_pts,)
        # xforce: (ntime_pts, nxforce)
        # returns: (ntime_pts, nstates)
        def dynamics(t: float, state: np.ndarray) -> np.ndarray:
            if state_postproc is not None:
                state = state_postproc(state)
            state_t = torch.as_tensor(state, dtype=torch.float32)[None, :]
            if xforce is None:
                dstates = self.forward(state_t, None)[0].squeeze(0)
            else:
                xforce_t = get_xforce_at_t(ts, xforce, t)
                dstates = self.forward(state_t, xforce_t)[0].squeeze(0)
            return dstates.detach().numpy()
        ret = scipy.integrate.solve_ivp(dynamics, [ts[0], ts[-1]], init_state, t_eval=ts, rtol=1e-9, atol=1e-9)
        states = ret.y.T
        return states

    def calc_loss(self, batch: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(batch) == 2:
            states, dstates = batch
            xforce = None
        elif len(batch) == 3:
            states, xforce, dstates = batch
        # dstates: (..., nstates), com: (..., ncom), com_jac: (..., nstates, ncom)
        with torch.enable_grad():
            dstates_pred, dstates_pred_ori, com, com_pred_jac = self.forward(states, xforce)
            loss0 = self._loss(dstates_pred, dstates)
            loss1 = self._loss(dstates_pred_ori, dstates)
            if com is not None:
                states_eps = states + torch.randn_like(states) * 0.1
                dstates_pred, dstates_pred_ori, com, com_pred_jac = self.forward(states_eps, xforce)
                reg = torch.mean(torch.einsum("...sc,...s->...c", com_pred_jac, dstates_pred_ori) ** 2)
            else:
                reg = 0
        return loss0, loss1, reg

    def training_step(self, train_batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        loss, loss_ori, reg = self.calc_loss(train_batch)
        self.log("training_loss", loss)
        self.log("training_loss_ori", loss_ori)
        self.log("training_reg", reg)
        return loss + loss_ori + reg

    def validation_step(self, val_batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        loss, loss_ori, reg = self.calc_loss(val_batch)
        self.log("val_loss", loss)
        self.log("val_loss_ori", loss_ori)
        self.log("val_reg", reg)
        return loss + loss_ori + reg

class NODE(pl.LightningModule):
    # this includes COMET & Neural ODE
    def __init__(self, nstates: int):
        super().__init__()
        self._nstates = nstates
        self._nn = MLP(nstates, nstates, activation=torch.nn.LogSigmoid, nhidden=250)
        self._loss = torch.nn.MSELoss()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        # returns the dynamics and the com
        dstates = self._nn(states)
        return dstates

    def time_series(self, init_state: np.ndarray, ts: np.ndarray) -> np.ndarray:
        # init_state: (nstates,)
        # ts: (ntime_pts,)
        # returns: (ntime_pts, nstates)
        def dynamics(t: float, state: np.ndarray) -> np.ndarray:
            dstates = self.forward(torch.as_tensor(state, dtype=torch.float32)[None, :]).squeeze(0)
            return dstates.detach().numpy()
        ret = scipy.integrate.solve_ivp(dynamics, [ts[0], ts[-1]], init_state, t_eval=ts, rtol=1e-9, atol=1e-9)
        states = ret.y.T
        return states

    def calc_loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        states, dstates = batch
        # dstates: (..., nstates)
        dstates_pred = self.forward(states)
        loss = self._loss(dstates_pred, dstates)
        return loss

    def training_step(self, train_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self.calc_loss(train_batch)
        self.log("training_loss", loss)
        return loss

    def validation_step(self, val_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self.calc_loss(val_batch)
        self.log("val_loss", loss)
        return loss

class HNN(pl.LightningModule):
    def __init__(self, nstates: int, nxforce: int = 0):
        super().__init__()
        assert nstates % 2 == 0
        self._nstates = nstates
        self._nxforce = nxforce
        self._nn = MLP(nstates + nxforce, 1, activation=torch.nn.LogSigmoid, nhidden=250)
        self._loss = torch.nn.MSELoss()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4)

    def forward(self, states: torch.Tensor, xforce: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # returns the dynamics and the hamiltonian
        states = states.requires_grad_()
        if xforce is None:
            xforce = torch.zeros((*states.shape[:-1], 0), dtype=states.dtype, device=states.device)
        inp = torch.cat((states, xforce), dim=-1)
        hamilt = self._nn(inp)  # (..., 1)
        dhamilt = torch.autograd.grad(hamilt, states, grad_outputs=torch.ones_like(hamilt), create_graph=True)[0]
        s2 = self._nstates // 2
        dstates = torch.cat((dhamilt[..., s2:], -dhamilt[..., :s2]), dim=-1)
        return dstates, hamilt
 
    def time_series(self, init_state: np.ndarray, ts: np.ndarray, xforce: Optional[np.ndarray] = None) -> np.ndarray:
        # init_state: (nstates,)
        # ts: (ntime_pts,)
        # returns: (ntime_pts, nstates)
        def dynamics(t: float, state: np.ndarray) -> np.ndarray:
            if xforce is None:
                dstates = self.forward(torch.as_tensor(state, dtype=torch.float32)[None, :], None)[0].squeeze(0)
            else:
                xforce_t = get_xforce_at_t(ts, xforce, t)
                dstates = self.forward(torch.as_tensor(state, dtype=torch.float32)[None, :], xforce_t)[0].squeeze(0)
            return dstates.detach().numpy()
        ret = scipy.integrate.solve_ivp(dynamics, [ts[0], ts[-1]], init_state, t_eval=ts, rtol=1e-9, atol=1e-9)
        states = ret.y.T
        return states

    def calc_loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if len(batch) == 2:
            states, dstates = batch
            xforce = None
        elif len(batch) == 3:
            states, xforce, dstates = batch
        # dstates: (..., nstates), com: (..., ncom), com_jac: (..., nstates, ncom)
        with torch.enable_grad():
            dstates_pred, _ = self.forward(states, xforce)
            loss = self._loss(dstates_pred, dstates)
        return loss

    def training_step(self, train_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self.calc_loss(train_batch)
        self.log("training_loss", loss)
        return loss

    def validation_step(self, val_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self.calc_loss(val_batch)
        self.log("val_loss", loss)
        return loss

class NSF(pl.LightningModule):
    def __init__(self, nstates: int, nxforce: int = 0):
        super().__init__()
        assert nstates % 2 == 0
        self._nn = MLP(nstates + nxforce, 1, activation=torch.nn.LogSigmoid, nhidden=250)
        self._ynn = MLP(nstates + nxforce, nstates, activation=torch.nn.LogSigmoid, nhidden=250)
        self._loss = torch.nn.MSELoss()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4)

    def forward(self, states: torch.Tensor, xforce: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # returns the dynamics and the hamiltonian
        states = states.requires_grad_()
        if xforce is None:
            xforce = torch.zeros((*states.shape[:-1], 0), dtype=states.dtype, device=states.device)
        def ynn_func(states: torch.Tensor, xforce: torch.Tensor) -> torch.Tensor:
            return self._ynn(torch.cat((states[None, :], xforce[None, :]), dim=-1)).squeeze(0)
        yjac = functorch.vmap(functorch.jacrev(ynn_func))(states, xforce)  # (..., nstates, nstates)
        w_mT = torch.linalg.inv(yjac - yjac.transpose(-2, -1))  # (..., nstates, nstates)
        hamilt = self._nn(torch.cat((states, xforce), dim=-1))  # (..., 1)
        dhamilt = torch.autograd.grad(hamilt, states, grad_outputs=torch.ones_like(hamilt), create_graph=True)[0]  # (..., nstates)
        dstates = torch.einsum("...ab,...b->...a", w_mT, dhamilt)
        return dstates, hamilt
 
    def time_series(self, init_state: np.ndarray, ts: np.ndarray, xforce: Optional[np.ndarray] = None) -> np.ndarray:
        # init_state: (nstates,)
        # ts: (ntime_pts,)
        # returns: (ntime_pts, nstates)
        def dynamics(t: float, state: np.ndarray) -> np.ndarray:
            if xforce is None:
                dstates = self.forward(torch.as_tensor(state, dtype=torch.float32)[None, :], None)[0].squeeze(0)
            else:
                xforce_t = get_xforce_at_t(ts, xforce, t)
                dstates = self.forward(torch.as_tensor(state, dtype=torch.float32)[None, :], xforce_t)[0].squeeze(0)
            return dstates.detach().numpy()
        ret = scipy.integrate.solve_ivp(dynamics, [ts[0], ts[-1]], init_state, t_eval=ts, rtol=1e-9, atol=1e-9)
        states = ret.y.T
        return states

    def calc_loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if len(batch) == 2:
            states, dstates = batch
            xforce = None
        elif len(batch) == 3:
            states, xforce, dstates = batch
        # dstates: (..., nstates), com: (..., ncom), com_jac: (..., nstates, ncom)
        with torch.enable_grad():
            dstates_pred, _ = self.forward(states, xforce)
            loss = self._loss(dstates_pred, dstates)
        return loss

    def training_step(self, train_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self.calc_loss(train_batch)
        self.log("training_loss", loss)
        return loss

    def validation_step(self, val_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self.calc_loss(val_batch)
        self.log("val_loss", loss)
        return loss

class LNN(pl.LightningModule):
    def __init__(self, nstates: int, nxforce: int):
        super().__init__()
        assert nstates % 2 == 0
        self._nn = MLP(nstates + nxforce, 1, activation=torch.nn.LogSigmoid, nhidden=250)
        self._loss = torch.nn.MSELoss()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4)

    def forward(self, states: torch.Tensor, xforce: Optional[torch.Tensor] = None) -> torch.Tensor:
        states = states.requires_grad_()

        if xforce is None:
            xforce = torch.zeros((*states.shape[:-1], 0), dtype=states.dtype, device=states.device)

        def get_lagrange(state: torch.Tensor, xforce: torch.Tensor):
            return self._nn(torch.cat((state[None, :], xforce[None, :]), dim=-1)).squeeze(0)

        n = states.shape[-1] // 2
        q, qt = torch.split(states, n, dim=-1)  # each has shape (..., nstates // 2)

        J = functorch.vmap(functorch.jacrev(get_lagrange))(states, xforce)  # (..., nout=1, nstates)
        # use nested jacrev here because forward AD in Hessian is not supported by logsigmoid
        H = functorch.vmap(functorch.jacrev(functorch.jacrev(get_lagrange)))(states, xforce)  # (..., nout=1, nstates, nstates)
        A = J[..., :n]  # (..., nout=1, nstates // 2)
        B = H[..., n:, n:]  # (..., nout=1, nstates // 2, nstates // 2)
        C = H[..., n:, :n]  # (..., nout=1, nstates // 2, nstates // 2)
        A = A.unsqueeze(-1)  # (..., nout=1, nstates // 2, 1)

        qtt = torch.linalg.pinv(B) @ (A - C @ qt[..., None, :, None])
        qtt = qtt.squeeze(-1).squeeze(-2)
        return torch.cat((qt, qtt), dim=-1)
    
    def time_series(self, init_state: np.ndarray, ts: np.ndarray, xforce: Optional[np.ndarray] = None) -> np.ndarray:
        # init_state: (nstates,)
        # ts: (ntime_pts,)
        # returns: (ntime_pts, nstates)
        def dynamics(t: float, state: np.ndarray) -> np.ndarray:
            if xforce is None:
                dstates = self.forward(torch.as_tensor(state, dtype=torch.float32)[None, :]).squeeze(0)
            else:
                xforce_t = get_xforce_at_t(ts, xforce, t)
                dstates = self.forward(torch.as_tensor(state, dtype=torch.float32)[None, :], xforce_t).squeeze(0)
            return dstates.detach().numpy()
        ret = scipy.integrate.solve_ivp(dynamics, [ts[0], ts[-1]], init_state, t_eval=ts, rtol=1e-9, atol=1e-9)
        states = ret.y.T
        return states

    def calc_loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if len(batch) == 2:
            states, dstates = batch
            xforce = None
        elif len(batch) == 3:
            states, xforce, dstates = batch
        # dstates: (..., nstates), com: (..., ncom), com_jac: (..., nstates, ncom)
        with torch.enable_grad():
            dstates_pred = self.forward(states, xforce)
            loss = self._loss(dstates_pred, dstates)
        return loss

    def training_step(self, train_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self.calc_loss(train_batch)
        self.log("training_loss", loss)
        return loss

    def validation_step(self, val_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self.calc_loss(val_batch)
        self.log("val_loss", loss)
        return loss

class COMET_Continuous(pl.LightningModule):
    def __init__(self, nstates: int, ncom: int, nhidden: int = 250) -> None:
        super().__init__()

        self._nn = torch.nn.Sequential(
            torch.nn.Conv1d(1, nhidden, kernel_size=5, padding=2, padding_mode="circular"),
            torch.nn.LogSigmoid(),
            torch.nn.Conv1d(nhidden, nhidden, kernel_size=5, padding=2, padding_mode="circular"),
            torch.nn.LogSigmoid(),
            torch.nn.Conv1d(nhidden, nhidden, kernel_size=5, padding=2, padding_mode="circular"),
            torch.nn.LogSigmoid(),
            torch.nn.Conv1d(nhidden, 1 + ncom, kernel_size=5, padding=2, padding_mode="circular"),
        )
        self._loss = torch.nn.MSELoss()
        self._dstates_nrm = 10.0
        self._ncom = ncom

    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # returns the dynamics and the com
        # states: (nbatch, nstates)
        states = states.requires_grad_()
        if self._ncom == 0:
            dstates = self._nn(states.unsqueeze(-2)).squeeze(-2)  # (nbatch, nstates)
            return dstates, dstates, None, None

        def _get_com_dstates(states):
            # states: (nstates,)
            nnout0 = self._nn(states.unsqueeze(0).unsqueeze(0))  # (1, 1 + ncom, nstates)
            nnout = nnout0.squeeze(0)
            dstates = nnout[0, :]  # (nstates,)
            com = torch.mean(nnout[1:, :], dim=-1)  # (ncom,)  # integral of each element
            return com, (dstates, com)
        jac_fcn = functorch.vmap(functorch.jacrev(_get_com_dstates, 0, has_aux=True))
        dcom_jac, (dstates, com) = jac_fcn(states)  # dcom_jac: (..., ncom, nstates), dstates: (..., nstates)
        dcom_jac = dcom_jac.transpose(-2, -1).contiguous()  # (..., nstates, ncom)
        dcom_jac_dstates = torch.cat((dcom_jac, dstates[..., None]), dim=-1)  # (..., nstates, ncom + 1)
        q, r = torch.linalg.qr(dcom_jac_dstates)  # q: (..., nstates, ncom + 1), r: (..., ncom + 1, ncom + 1)
        dstates_ortho = q[..., -1] * r[..., -1, -1:]  # (..., nstates)
        return dstates_ortho, dstates, com, dcom_jac

    def time_series(self, init_state: np.ndarray, ts: np.ndarray) -> np.ndarray:
        # init_state: (nstates,)
        # ts: (ntime_pts,)
        # returns: (ntime_pts, nstates)
        def dynamics(t: float, state: np.ndarray) -> np.ndarray:
            dstates = self.forward(torch.as_tensor(state, dtype=torch.float32)[None, :])[0].squeeze(0)
            return dstates.detach().numpy() * self._dstates_nrm
        ret = scipy.integrate.solve_ivp(dynamics, [ts[0], ts[-1]], init_state, t_eval=ts, rtol=1e-9, atol=1e-9)
        states = ret.y.T
        return states

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4)

    def calc_loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        states, dstates = batch
        # dstates: (..., nstates), com: (..., ncom), com_jac: (..., nstates, ncom)
        with torch.enable_grad():
            dstates_pred, dstates_pred_ori, com, com_pred_jac = self.forward(states)
            # print(dstates_pred[0], dstates_pred_ori[0], states[0], dstates[0])
            loss0 = self._loss(dstates_pred, dstates / self._dstates_nrm)
            loss1 = self._loss(dstates_pred_ori, dstates / self._dstates_nrm)
            if com is not None:
                states_eps = states + torch.randn_like(states) * 0.1
                dstates_pred, dstates_pred_ori, com, com_pred_jac = self.forward(states_eps)
                reg = torch.mean(torch.einsum("...sc,...s->...c", com_pred_jac, dstates_pred_ori) ** 2)
            else:
                reg = 0
        return loss0, loss1, reg

    def training_step(self, train_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        # train_batch: (nbatch, npts) represents u, (nbatch, npts) represents \dot{u}
        loss, loss_ori, reg = self.calc_loss(train_batch)
        self.log("training_loss", loss)
        self.log("training_loss_ori", loss_ori)
        self.log("training_reg", reg)
        return loss + loss_ori + reg

    def validation_step(self, val_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, loss_ori, reg = self.calc_loss(val_batch)
        self.log("val_loss", loss)
        self.log("val_loss_ori", loss_ori)
        self.log("val_reg", reg)
        return loss + loss_ori + reg

def get_xforce_at_t(ts: np.ndarray, xforce: np.ndarray, t: float) -> torch.Tensor:
    tidx = np.searchsorted(ts, t)
    frac = (t - ts[tidx - 1]) / (ts[tidx] - ts[tidx - 1])
    xforce_t = torch.as_tensor(xforce[tidx - 1] + (xforce[tidx] - xforce[tidx - 1]) * frac, dtype=torch.float32)[None, :]
    return xforce_t

def get_module(method: str, nstates: int, ncom: int, nxforce: int = 0) -> pl.LightningModule:
    if method == "comet":
        module = CoMet(nstates=nstates, ncom=ncom, nxforce=nxforce)
    elif method.startswith("comet-"):
        module = CoMet(nstates=nstates, ncom=ncom, nhidden=int(method[6:]))
    elif method == "cometcont":
        module = COMET_Continuous(nstates=nstates, ncom=ncom)
    elif method == "node":
        module = NODE(nstates=nstates)
    elif method == "hnn":
        module = HNN(nstates=nstates, nxforce=nxforce)
    elif method == "nsf":
        module = NSF(nstates=nstates, nxforce=nxforce)
    elif method == "lnn":
        module = LNN(nstates=nstates, nxforce=nxforce)
    else:
        raise RuntimeError(f"Unknown method: {method}")
    return module

def load_module(method: str, ckpt_path: str, nstates: int, ncom: int, nxforce: int = 0) -> pl.LightningModule:
    if method == "comet":
        nn = CoMet.load_from_checkpoint(ckpt_path, nstates=nstates, ncom=ncom, nxforce=nxforce)
    elif method.startswith("comet-"):
        nn = CoMet.load_from_checkpoint(ckpt_path, nstates=nstates, ncom=ncom, nxforce=nxforce, nhidden=int(method[6:]))
    elif method == "cometcont":
        nn = COMET_Continuous.load_from_checkpoint(ckpt_path, nstates=nstates, ncom=ncom)
    elif method == "node":
        nn = NODE.load_from_checkpoint(ckpt_path, nstates=nstates)
    elif method == "hnn":
        nn = HNN.load_from_checkpoint(ckpt_path, nstates=nstates, nxforce=nxforce)
    elif method == "nsf":
        nn = NSF.load_from_checkpoint(ckpt_path, nstates=nstates, nxforce=nxforce)
    elif method == "lnn":
        nn = LNN.load_from_checkpoint(ckpt_path, nstates=nstates, nxforce=nxforce)
    else:
        raise RuntimeError(f"Unknown method: {method}")
    return nn
