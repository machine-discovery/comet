from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import scipy.integrate


@dataclass
class BaseSim:
    nstates: int
    ncom: int
    noise: float
    name: str
    nxforce: int = 0

    def simulate(self, *args):
        pass

    def get_params_list(self, ndset: int) -> List[np.ndarray]:
        pass

    def get_coms(self) -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
        pass

    def postproc(self, states: np.ndarray) -> np.ndarray:
        return states

class MassSpring(BaseSim):
    def __init__(self, noise: float = 0.05):
        super().__init__(nstates=2, ncom=1, noise=noise, name="Mass-spring")

    def simulate(self, x0: float, v0: float, ts: np.ndarray) -> np.ndarray:
        def dynamics(t: float, state: np.ndarray) -> np.ndarray:
            return np.stack((state[..., 1], -state[..., 0]), axis=-1)
        ret = scipy.integrate.solve_ivp(dynamics, [ts[0], ts[-1]], np.asarray([x0, v0]), t_eval=ts, rtol=1e-9, atol=1e-9)
        states = np.asarray(ret.y).T  # (nt, nstates)
        dstates = dynamics(0, states)  # (nt, nstates)
        dstates = dstates + np.random.randn(*dstates.shape) * self.noise
        states_dstates = np.concatenate((states, dstates), axis=-1)
        return states_dstates

    def get_params_list(self, ndset: int) -> List[np.ndarray]:
        x0_list = np.random.rand(ndset) * 1 - 0.5
        v0_list = np.random.rand(ndset) * 1 - 0.5
        return [x0_list, v0_list]

    def get_coms(self) -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
        energy = lambda states: 0.5 * (states[..., 0] ** 2 + states[..., 1] ** 2)
        return {"Energy": energy}

class Pendulum2D(BaseSim):
    def __init__(self, noise: float = 0.05):
        super().__init__(nstates=4, ncom=3, noise=noise, name="2D pendulum")

    def simulate(self, theta0: float, omega0: float, ts: np.ndarray) -> np.ndarray:
        def dynamics(t: float, state: np.ndarray) -> np.ndarray:
            return np.stack([state[..., 1], -np.sin(state[..., 0])], axis=-1)
        ret = scipy.integrate.solve_ivp(dynamics, [ts[0], ts[-1]], np.asarray([theta0, omega0]), t_eval=ts, rtol=1e-9, atol=1e-9)
        theta, omega = ret.y
        x = np.sin(theta)  # (nt,)
        y = -np.cos(theta)
        dx = omega * np.cos(theta)
        dy = omega * np.sin(theta)
        ddx = -omega ** 2 * np.sin(theta) - np.sin(theta) * np.cos(theta)
        ddy = omega ** 2 * np.cos(theta) - np.sin(theta) ** 2
        states = np.stack((x, y, dx, dy), axis=-1)  # (nt, nstates)
        dstates = np.stack((dx, dy, ddx, ddy), axis=-1)  # (nt, nstates)
        dstates = dstates + np.random.randn(*dstates.shape) * self.noise
        states_dstates = np.concatenate((states, dstates), axis=-1)
        return states_dstates

    def get_params_list(self, ndset: int) -> List[np.ndarray]:
        theta0_list = np.random.rand(ndset) * 2 - 1
        omega0_list = np.random.rand(ndset) * 2 - 1
        return [theta0_list, omega0_list]

    def get_coms(self) -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
        energy = lambda states: states[..., 1] + 0.5 * (states[..., 2] ** 2 + states[..., 3] ** 2)
        length = lambda states: states[..., 0] ** 2 + states[..., 1] ** 2
        angle = lambda states: states[..., 0] * states[..., 2] + states[..., 1] * states[..., 3]
        return {"Energy": energy, "Length": length, "Velocity angle": angle}

class ForcedPendulum2D(BaseSim):
    def __init__(self, noise: float = 0.05):
        super().__init__(nstates=4, ncom=3, nxforce=1, noise=noise, name="2D pendulum")

    def simulate(self, theta0: float, omega0: float, xtorque: Callable, ts: np.ndarray) -> np.ndarray:
        def dynamics(t: float, state: np.ndarray) -> np.ndarray:
            xtorque_t = xtorque(t)
            return np.stack([state[..., 1], -np.sin(state[..., 0]) + xtorque_t * np.cos(state[..., 0])], axis=-1)
        ret = scipy.integrate.solve_ivp(dynamics, [ts[0], ts[-1]], np.asarray([theta0, omega0]), t_eval=ts, rtol=1e-9, atol=1e-9)
        theta, omega = ret.y
        x = np.sin(theta)  # (nt,)
        y = -np.cos(theta)
        dx = omega * np.cos(theta)
        dy = omega * np.sin(theta)
        ddx = -omega ** 2 * np.sin(theta) - np.sin(theta) * np.cos(theta)
        ddy = omega ** 2 * np.cos(theta) - np.sin(theta) ** 2
        states = np.stack((x, y, dx, dy), axis=-1)  # (nt, nstates)
        dstates = np.stack((dx, dy, ddx, ddy), axis=-1)  # (nt, nstates)
        dstates = dstates + np.random.randn(*dstates.shape) * self.noise
        states_xtorque_dstates = np.concatenate(
            (states, xtorque(ts)[:, None], dstates), axis=-1)
        return states_xtorque_dstates

    def get_params_list(self, ndset: int) -> List[np.ndarray]:
        theta0_list = np.random.rand(ndset) * 2 - 1
        omega0_list = np.random.rand(ndset) * 2 - 1

        ampls = np.random.rand(ndset) * 1 - 0.5
        omegas = np.random.randn(ndset) * 5
        phases = np.random.rand(ndset) * 2 * np.pi
        xtorque_list = [lambda t: ampl * np.cos(omega * t + phase) for (ampl, omega, phase) in zip(ampls, omegas, phases)]
        return [theta0_list, omega0_list, xtorque_list]

    def get_coms(self) -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
        energy = lambda states, nxforce: states[..., 1] + 0.5 * (states[..., 2] ** 2 + states[..., 3] ** 2) - nxforce[..., 0] * states[..., 0]
        length = lambda states, _: states[..., 0] ** 2 + states[..., 1] ** 2
        angle = lambda states, _: states[..., 0] * states[..., 2] + states[..., 1] * states[..., 3]
        return {"Energy": energy, "Length": length, "Velocity angle": angle}

class DampedPendulum2D(BaseSim):
    def __init__(self, noise: float = 0.05):
        super().__init__(nstates=4, ncom=2, noise=noise, name="Damped pendulum")

    def simulate(self, theta0: float, omega0: float, ts: np.ndarray) -> np.ndarray:
        def dynamics(t: float, state: np.ndarray) -> np.ndarray:
            return np.stack([state[..., 1], -np.sin(state[..., 0]) - state[..., 1]], axis=-1)
        ret = scipy.integrate.solve_ivp(dynamics, [ts[0], ts[-1]], np.asarray([theta0, omega0]), t_eval=ts, rtol=1e-9, atol=1e-9)
        theta, omega = ret.y
        x = np.sin(theta)  # (nt,)
        y = -np.cos(theta)
        dx = omega * np.cos(theta)
        dy = omega * np.sin(theta)
        ddx = -omega ** 2 * np.sin(theta) - np.sin(theta) * np.cos(theta) - dx
        ddy = omega ** 2 * np.cos(theta) - np.sin(theta) ** 2 - dy
        states = np.stack((x, y, dx, dy), axis=-1)  # (nt, nstates)
        dstates = np.stack((dx, dy, ddx, ddy), axis=-1)  # (nt, nstates)
        dstates = dstates + np.random.randn(*dstates.shape) * self.noise
        states_dstates = np.concatenate((states, dstates), axis=-1)
        return states_dstates

    def get_params_list(self, ndset: int) -> List[np.ndarray]:
        theta0_list = np.random.rand(ndset) * 2 - 1
        omega0_list = np.random.rand(ndset) * 2 - 1
        return [theta0_list, omega0_list]

    def get_coms(self) -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
        length = lambda states: states[..., 0] ** 2 + states[..., 1] ** 2
        angle = lambda states: states[..., 0] * states[..., 2] + states[..., 1] * states[..., 3]
        return {"Length": length, "Velocity angle": angle}

class NonlinSpring3D(BaseSim):
    def __init__(self, noise: float = 0.05):
        super().__init__(nstates=6, ncom=4, noise=noise, name="Nonlinear spring (3D)")
        self._power = 2.0

    def simulate(self, x0: float, y0: float, z0: float, vx0: float, vy0: float, vz0: float, ts: np.ndarray) -> np.ndarray:
        def dynamics(t: float, state: np.ndarray) -> np.ndarray:
            pos = state[..., :3]  # (..., 3)
            dist = np.sum(pos ** 2, axis=-1, keepdims=True) ** .5
            acc = -(dist ** self._power) * pos  # (..., 3)
            return np.concatenate((state[..., 3:], acc), axis=-1)
        ret = scipy.integrate.solve_ivp(dynamics, [ts[0], ts[-1]], np.asarray([x0, y0, z0, vx0, vy0, vz0]), t_eval=ts, rtol=1e-9, atol=1e-9)
        states = np.asarray(ret.y).T  # (nt, nstates)
        dstates = dynamics(0, states)  # (nt, nstates)
        dstates = dstates + np.random.randn(*dstates.shape) * self.noise
        states_dstates = np.concatenate((states, dstates), axis=-1)
        return states_dstates

    def get_params_list(self, ndset: int) -> List[np.ndarray]:
        x0_list = np.random.rand(ndset) * 2 - 1
        y0_list = np.random.rand(ndset) * 2 - 1
        z0_list = np.random.rand(ndset) * 2 - 1
        vx0_list = np.random.rand(ndset) * 2 - 1
        vy0_list = np.random.rand(ndset) * 2 - 1
        vz0_list = np.random.rand(ndset) * 2 - 1
        return [x0_list, y0_list, z0_list, vx0_list, vy0_list, vz0_list]

class NonlinSpring2D(BaseSim):
    def __init__(self, noise: float = 0.05):
        super().__init__(nstates=4, ncom=2, noise=noise, name="Nonlinear spring (2D)")  # energy and angmom
        self._power = 2.0

    def simulate(self, x0: float, y0: float, vx0: float, vy0: float, ts: np.ndarray) -> np.ndarray:
        def dynamics(t: float, state: np.ndarray) -> np.ndarray:
            pos = state[..., :2]  # (..., 2)
            dist = np.sum(pos ** 2, axis=-1, keepdims=True) ** .5
            acc = -(dist ** self._power) * pos  # (..., 2)
            return np.concatenate((state[..., 2:], acc), axis=-1)
        ret = scipy.integrate.solve_ivp(dynamics, [ts[0], ts[-1]], np.asarray([x0, y0, vx0, vy0]), t_eval=ts, rtol=1e-9, atol=1e-9)
        states = np.asarray(ret.y).T  # (nt, nstates)
        dstates = dynamics(0, states)  # (nt, nstates)
        dstates = dstates + np.random.randn(*dstates.shape) * self.noise
        states_dstates = np.concatenate((states, dstates), axis=-1)
        return states_dstates

    def get_params_list(self, ndset: int) -> List[np.ndarray]:
        x0_list = np.random.rand(ndset) * 2 - 1
        y0_list = np.random.rand(ndset) * 2 - 1
        vx0_list = np.random.rand(ndset) * 2 - 1
        vy0_list = np.random.rand(ndset) * 2 - 1
        return [x0_list, y0_list, vx0_list, vy0_list]

    def get_coms(self) -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
        energy = lambda states: 0.5 * (states[..., 2] ** 2 + states[..., 3] ** 2) + ((states[..., 0] ** 2 + states[..., 1] ** 2) ** 0.5) ** (self._power + 2) / (self._power + 2)
        angmom = lambda states: states[..., 0] * states[..., 3] - states[..., 1] * states[..., 2]
        return {"Energy": energy, "Angular momentum": angmom}

class TwoBody(BaseSim):
    def __init__(self, noise: float = 0.05):
        super().__init__(nstates=8, ncom=7, noise=noise, name="Two body")

    def simulate(self, r0: float, theta0: float, v0: float, ts: np.ndarray) -> np.ndarray:
        def dynamics(t: float, state: np.ndarray) -> np.ndarray:
            # state: (x1, y1, x2, y2, vx1, vy1, vx2, vy2)
            r1, r2 = state[..., :2], state[..., 2:4]  # (..., 2)
            d12_32_inv = (np.sum((r1 - r2) ** 2, axis=-1, keepdims=True)) ** (-1.5)
            a1 = d12_32_inv * (r2 - r1)
            a2 = -a1
            dstate = np.concatenate((state[..., 4:], a1, a2), axis=-1)
            return dstate
        x0, y0 = r0 * np.cos(theta0), r0 * np.sin(theta0)
        state0 = np.asarray([x0, y0, -x0, -y0, -v0 * np.sin(theta0), v0 * np.cos(theta0), v0 * np.sin(theta0), -v0 * np.cos(theta0)])
        ret = scipy.integrate.solve_ivp(dynamics, [ts[0], ts[-1]], state0, t_eval=ts, rtol=1e-9, atol=1e-9)
        states = np.asarray(ret.y).T  # (nt, nstates)
        dstates = dynamics(0, states)  # (nt, nstates)
        dstates = dstates + np.random.randn(*dstates.shape) * self.noise
        states_dstates = np.concatenate((states, dstates), axis=-1)
        return states_dstates

    def get_params_list(self, ndset: int) -> List[np.ndarray]:
        r0_list = np.random.rand(ndset) * 1 + 0.5
        theta0_list = np.random.rand(ndset) * (2 * np.pi) - np.pi
        v0_list = (np.random.rand(ndset) * 0.3 + 0.7) * (0.5 / r0_list ** 0.5)
        return [r0_list, theta0_list, v0_list]

    def get_coms(self) -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
        energy = lambda states: 0.5 * np.sum(states[..., 4:8] ** 2, axis=-1) - 1.0 / np.sqrt((states[..., 0] - states[..., 2]) ** 2 + (states[..., 1] - states[..., 3]) ** 2)
        xmom = lambda states: states[..., 4] + states[..., 6]
        ymom = lambda states: states[..., 5] + states[..., 7]
        angmom = lambda states: (states[..., 0] * states[..., 5] - states[..., 1] * states[..., 4]) + (states[..., 2] * states[..., 7] - states[..., 3] * states[..., 6])
        return {"Energy": energy, "x-momentum": xmom, "y-momentum": ymom, "Angular momentum": angmom}

class LotkaVolterra(BaseSim):
    def __init__(self, noise: float = 0.05):
        super().__init__(nstates=2, ncom=1, noise=noise, name="Lotka-Volterra")

    def simulate(self, x0: float, y0: float, ts: np.ndarray) -> np.ndarray:
        def dynamics(t: float, state: np.ndarray) -> np.ndarray:
            x, y = np.split(state, 2, axis=-1)
            dx = x - x * y
            dy = -y + x * y
            return np.concatenate((dx, dy), axis=-1)
        ret = scipy.integrate.solve_ivp(dynamics, [ts[0], ts[-1]], np.asarray([x0, y0]), t_eval=ts, rtol=1e-9, atol=1e-9)
        states = np.asarray(ret.y).T  # (nt, nstates)
        dstates = dynamics(0, states)  # (nt, nstates)
        dstates = dstates + np.random.randn(*dstates.shape) * self.noise
        states_dstates = np.concatenate((states, dstates), axis=-1)
        return states_dstates

    def get_params_list(self, ndset: int) -> List[np.ndarray]:
        x0_list = np.random.rand(ndset) * 1.5 + 0.5
        y0_list = np.random.rand(ndset) * 1.5 + 0.5
        return [x0_list, y0_list]

    def get_coms(self) -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
        v = lambda states: -states[..., 0] + np.log(states[..., 0]) - states[..., 1] + np.log(states[..., 1])
        return {"V": v}

    def get_states_name(self) -> List[str]:
        return ["Prey", "Predator"]

class KDVCont(BaseSim):
    def __init__(self, noise: float = 0.05):
        super().__init__(nstates=100, ncom=3, noise=noise, name="KdV equation")
        self._length = 5.0
        self._delta = 0.00022
        self._normalize = lambda phi: (phi - 1) / 2.0
        self._denormalize = lambda state: (state * 2.0) + 1.0
        self._normalize_dt = lambda dphi: dphi / 2.0

    def simulate(self, init_state: np.ndarray, ts: np.ndarray) -> np.ndarray:
        # init_state: (nstates,)
        def dynamics(t: float, state: np.ndarray) -> np.ndarray:
            # state: (..., nstates)
            if np.any(np.isnan(state)):
                assert False
            dh = self._length / state.shape[-1]
            phi = self._denormalize(state)
            phi_p2 = np.roll(phi, shift=-2, axis=-1)
            phi_p1 = np.roll(phi, shift=-1, axis=-1)
            phi_m1 = np.roll(phi, shift=1, axis=-1)
            phi_m2 = np.roll(phi, shift=2, axis=-1)
            d3x_phi = (phi_p2 - 2 * phi_p1 + 2 * phi_m1 - phi_m2) / (2 * dh ** 3)
            dx_phi = (phi_p1 - phi_m1) / (2 * dh)
            dphi = (phi_p1 + phi + phi_m1) / 3.0 * dx_phi - self._delta ** 2 * d3x_phi
            dstate = self._normalize_dt(dphi)
            return dstate
        ret = scipy.integrate.solve_ivp(dynamics, [ts[0], ts[-1]], init_state, t_eval=ts, rtol=1e-9, atol=1e-9)
        states = np.asarray(ret.y).T  # (nt, nstates)
        dstates = dynamics(0, states)  # (nt, nstates)
        dstates = dstates + np.random.randn(*dstates.shape) * self.noise
        states = self.postproc(states)
        states_dstates = np.concatenate((states, dstates), axis=-1)
        return states_dstates

    def get_params_list(self, ndset: int) -> List[np.ndarray]:
        x = np.linspace(0, self._length, self.nstates + 1)[:self.nstates]  # (nstates)
        phase = np.random.rand(ndset)[:, None] * (2 * np.pi)  # (ndset, 1)
        offset = np.random.rand(ndset)[:, None] + 1.5  # (ndset, 1)
        ampl = np.random.rand(ndset)[:, None]  # (ndset, 1)
        freq = np.random.randint(1, 2, size=(ndset, 1))
        init_phi0 = offset + ampl * np.cos(2 * np.pi / self._length * freq * x + phase)  # (ndset, nstates + 1)
        init_states = self._normalize(init_phi0)
        return [init_states]
    
    def get_coms(self) -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
        def diff(x: np.ndarray):
            dh = self._length / x.shape[-1]
            xp1 = np.roll(x, shift=-1, axis=-1)
            xm1 = np.roll(x, shift=1, axis=-1)
            return (xp1 - xm1) / (2 * dh)
        mass = lambda states: np.mean(self._denormalize(states), axis=-1)
        momentum = lambda states: np.mean(self._denormalize(states) ** 2, axis=-1)
        energy = lambda states: np.mean(self._denormalize(states) ** 3 - diff(self._denormalize(states)) ** 2 * self._delta * 15, axis=-1)
        energy1 = lambda states: np.mean(self._denormalize(states) ** 3, axis=-1)
        energy2 = lambda states: np.mean(diff(self._denormalize(states)) ** 2, axis=-1)
        return {"Mass": mass, "Momentum": momentum, "Energy": energy, "Energy1": energy1, "Energy2": energy2}

def get_sim(sim: str, noise: float = 0.05) -> BaseSim:
    if sim == "mass-spring":
        return MassSpring(noise)
    elif sim == "2d-pendulum":
        return Pendulum2D(noise)
    elif sim == "forced-2d-pendulum":
        return ForcedPendulum2D(noise)
    elif sim == "damped-pendulum":
        return DampedPendulum2D(noise)
    elif sim == "nonlin-spring":
        return NonlinSpring3D(noise)
    elif sim == "nonlin-spring-2d":
        return NonlinSpring2D(noise)
    elif sim == "two-body":
        return TwoBody(noise)
    elif sim == "lotka-volterra":
        return LotkaVolterra(noise)
    elif sim == "kdvcont":
        return KDVCont(noise)
    else:
        avails = ["mass-spring", "forced-2d-pendulum", "2d-pendulum", "damped-pendulum", "nonlin-spring", "nonlin-spring-2d", "two-body"]
        raise RuntimeError(f"Unknown sim: {sim}. Options are: {avails}")

if __name__ == "__main__":
    sim = KDVCont(noise=0)
    x = np.linspace(0, 2, sim.nstates // 2 + 1)[:sim.nstates // 2]
    init_state = sim.get_params_list(1)[0][0]
    ts = np.linspace(0, 10, 100)
    states_dstates = sim.simulate(init_state, ts)
    print("Simulation done")
    states = states_dstates[..., :sim.nstates]  # (nt, nstates)

    from matplotlib import pyplot as plt
    from matplotlib import animation

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 2), ylim=(0, 5))
    line, = ax.plot([], [], lw=2)
    def init():
        line.set_data([], [])
        return line,
    
    def animate(i):
        x = np.linspace(0, 2, states.shape[-1])
        y = states[i]
        line.set_data(x, y)
        return line,
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=states.shape[0], interval=20, blit=True)
    anim.save('animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
