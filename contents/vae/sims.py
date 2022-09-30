from typing import List, Tuple
from dataclasses import dataclass
import torch
import numpy as np
import scipy.integrate


@dataclass
class BaseVAESim:
    data_shape: Tuple[int, int, int]
    dt: float
    name: str

    def simulate(self, *args):
        pass

    def get_params_list(self, ndset: int) -> List[np.ndarray]:
        pass

    def to_imshow(self, data: torch.Tensor) -> np.ndarray:
        # data has the shape of self.data_shape, returns the numpy array for plt.imshow
        pass

class VAETwoBody(BaseVAESim):
    def __init__(self, dt: float = 1e-2):
        super().__init__(data_shape=(2, 30, 30), dt=dt, name="Two body")

    def simulate(self, r0: float, theta0: float, v0: float, ts: np.ndarray) -> np.ndarray:
        def dynamics(t: float, state: np.ndarray) -> np.ndarray:
            # state: (x1, y1, x2, y2, vx1, vy1, vx2, vy2)
            r1, r2 = state[..., :2], state[..., 2:4]  # (..., 2)
            d12_32_inv = (np.sum((r1 - r2) ** 2, axis=-1, keepdims=True)) ** (-1.5)
            a1 = d12_32_inv * (r2 - r1)
            a2 = -a1
            dstate = np.concatenate((state[..., 4:], a1, a2), axis=-1)
            return dstate
        
        # get the time steps to be evaluated
        ts1 = ts + self.dt  # (nt0,)
        ts2 = ts + self.dt * 2
        ts_tot = np.stack((ts, ts1, ts2), axis=-1)  # (nt0, 3)
        ts_tot = np.reshape(ts_tot, -1)  # (nt=nt0 * 3)

        # get the states
        x0, y0 = r0 * np.cos(theta0), r0 * np.sin(theta0)
        state0 = np.asarray([x0, y0, -x0, -y0, -v0 * np.sin(theta0), v0 * np.cos(theta0), v0 * np.sin(theta0), -v0 * np.cos(theta0)])
        ret = scipy.integrate.solve_ivp(dynamics, [ts_tot[0], ts_tot[-1]], state0, t_eval=ts_tot, rtol=1e-9, atol=1e-9)
        states = np.asarray(ret.y).T  # (nt, nstates)

        # draw the images from the states
        # xg, yg: (height, width)
        xg, yg = np.meshgrid(np.linspace(-1.7, 1.7, self.data_shape[-1]), np.linspace(-1.7, 1.7, self.data_shape[-2]), indexing="xy")
        st = states[..., None, None]  # (nt, nstates, 1, 1)
        size = 0.2
        ch1 = np.exp(-((xg - st[:, 0]) ** 2 + (yg - st[:, 1]) ** 2) / (2 * size ** 2))  # (nt, height, width)
        ch2 = np.exp(-((xg - st[:, 2]) ** 2 + (yg - st[:, 3]) ** 2) / (2 * size ** 2))  # (nt, height, width)
        imgs = np.stack((ch1, ch2), axis=1)  # (nt, nchannels, height, width)
        imgs = np.reshape(imgs, (-1, 3, *self.data_shape))  # (nt0, 3, nchannels, height, width)
        return imgs

    def get_params_list(self, ndset: int) -> List[np.ndarray]:
        r0_list = np.random.rand(ndset) * 1 + 0.5
        theta0_list = np.random.rand(ndset) * (2 * np.pi) - np.pi
        v0_list = (np.random.rand(ndset) * 0.3 + 0.7) * (0.5 / r0_list ** 0.5)
        return [r0_list, theta0_list, v0_list]
    
    def to_imshow(self, data: torch.Tensor) -> np.ndarray:
        # data: (nchannels, height, width)
        data2 = torch.movedim(data.detach().cpu(), -3, -1)  # (height, width, nchannels)
        img = torch.zeros((data2.size(0), data2.size(1), 3), dtype=data2.dtype, device=data2.device)
        img[..., :2] = data2
        return img.detach().cpu().numpy()

class VAEPendulum2D(BaseVAESim):
    def __init__(self, dt: float = 1e-2):
        super().__init__(data_shape=(1, 30, 30), dt=dt, name="2D pendulum")

    def simulate(self, theta0: float, omega0: float, ts: np.ndarray) -> np.ndarray:
        def dynamics(t: float, state: np.ndarray) -> np.ndarray:
            return np.stack([state[..., 1], -np.sin(state[..., 0])], axis=-1)

        # get the time steps to be evaluated
        ts1 = ts + self.dt  # (nt0,)
        ts2 = ts + self.dt * 2
        ts_tot = np.stack((ts, ts1, ts2), axis=-1)  # (nt0, 3)
        ts_tot = np.reshape(ts_tot, -1)  # (nt=nt0 * 3)

        ret = scipy.integrate.solve_ivp(dynamics, [ts_tot[0], ts_tot[-1]], np.asarray([theta0, omega0]), t_eval=ts_tot, rtol=1e-9, atol=1e-9)
        theta, omega = ret.y
        x = np.sin(theta)  # (nt,)
        y = -np.cos(theta)

        # draw the images from the states
        # xg, yg: (height, width)
        xg, yg = np.meshgrid(np.linspace(-1.2, 1.2, self.data_shape[-1]), np.linspace(-1.2, 1.2, self.data_shape[-2]), indexing="xy")
        xt = x[..., None, None]  # (nt, 1, 1)
        yt = y[..., None, None]  # (nt, 1, 1)
        size = 0.2
        ch1 = np.exp(-((xg - xt) ** 2 + (yg - yt) ** 2) / (2 * size ** 2))  # (nt, height, width)
        imgs = ch1[:, None, :, :]  # (nt, nchannels, height, width)
        imgs = np.reshape(imgs, (-1, 3, *self.data_shape))  # (nt0, 3, nchannels, height, width)
        return imgs

    def get_params_list(self, ndset: int) -> List[np.ndarray]:
        theta0_list = np.random.rand(ndset) * 2 - 1
        omega0_list = np.random.rand(ndset) * 2 - 1
        return [theta0_list, omega0_list]

    def to_imshow(self, data: torch.Tensor) -> np.ndarray:
        # data: (nchannels, height, width)
        data2 = torch.movedim(data.detach().cpu(), -3, -1)  # (height, width, nchannels)
        img = torch.zeros((data2.size(0), data2.size(1), 3), dtype=data2.dtype, device=data2.device)
        img[..., 1:2] = data2
        return img.detach().cpu().numpy()

def get_sim(sim: str, dt: float = 0.01) -> BaseVAESim:
    if sim == "two-body":
        return VAETwoBody(dt=dt)
    elif sim == "2d-pendulum":
        return VAEPendulum2D(dt=dt)
    else:
        avails = ["two-body"]
        raise RuntimeError(f"Unknown sim: {sim}. Options are: {avails}")

if __name__ == "__main__":
    sim = VAETwoBody()
    params_list = [p[0] for p in sim.get_params_list(1)]
    ts = np.linspace(0, 10, 100)
    imgs = sim.simulate(*params_list, ts)

    import matplotlib.pyplot as plt
    imgs = np.sum(imgs, axis=-3)
    plt.figure(figsize=(12, 3))
    for i in range(3):
        plt.subplot(1, 4, i + 1)
        plt.imshow(imgs[0, i])
        plt.colorbar()
    plt.subplot(1, 4, 4)
    plt.imshow(imgs[0, 0] - imgs[0, 1])
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("fig.png")
    plt.close()
