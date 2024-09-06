import torch
from torch import nn
from jaxtyping import Array, Float, Integer, Complex
from typing import Callable
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotly.graph_objects as go

type HilbertVector = Complex[torch.Tensor, "num_walkers"] | Float[
    torch.Tensor, "num_walkers"
]
type Psi = Callable[
    [Float[torch.Tensor, "3 num_walkers"]],
    Complex[torch.Tensor, "num_walkers"] | Float[torch.Tensor, "num_walkers"],
]
type Position = Float[torch.Tensor, "3 #num_walkers"]

type E_loc = Float[torch.Tensor, "num_walkers"] | Complex[torch.Tensor, "num_walkers"]

type PositionFunction = Callable[
    [Position, E_loc],
    Float[torch.Tensor, "num_walkers"],
]

type PositionFunctionBuffer = Complex[
    torch.Tensor, "n_pos_functions, n_walkers"
] | Float[torch.Tensor, "n_pos_functions, n_walkers"]


class MarkovChain:
    def __init__(
        self,
        psi: Psi,
        r_min: Position,
        r_max: Position,
        n_walkers: int,
    ):
        self.state: Position = torch.rand(3, n_walkers)

        # constants
        self.psi: Psi = psi
        self.n_walkers: int = n_walkers
        self.r_min: Position = r_min.reshape(3, 1)
        self.r_max: Position = r_max.reshape(3, 1)

    def make_step(self, step_size: float):
        step: Position = torch.randn(3, self.n_walkers)
        step = step * (self.r_max - self.r_min) * step_size
        return step

    def E_loc(self, position_samples: Position) -> Float[torch.Tensor, "num_walkers"]:
        theta = self.psi.theta
        alpha = theta[0]

        r = torch.norm(position_samples, dim=0)

        return -0.5 * alpha * (alpha - 2 / r) - 1 / r

    def d_ln_psi(
        self,
        position_samples: Position,
    ) -> Float[torch.Tensor, "num_walkers"]:
        theta = self.psi.theta
        alpha = theta[0]

        r = torch.norm(position_samples, dim=0)

        return -(r**2)

    def metropolis(self, step_size: float, calc_grad_terms: bool = False):
        proposed_step: Position = self.make_step(step_size)
        proposed_state: Position = self.state + proposed_step

        prob_dist = self.psi(self.state)
        prob_dist = torch.real(prob_dist.conj() * prob_dist)

        proposed_prob_dist = self.psi(proposed_state)
        proposed_prob_dist = torch.real(proposed_prob_dist.conj() * proposed_prob_dist)

        # TODO: does this min condition make sense?
        acceptance_mask = torch.rand(self.n_walkers) < torch.min(
            proposed_prob_dist / prob_dist, torch.tensor(1.0)
        )

        self.state = torch.where(acceptance_mask, proposed_state, self.state)

        if calc_grad_terms:
            E_loc_i = torch.mean(self.E_loc(self.state))
            d_ln_psi_i = torch.mean(self.d_ln_psi(self.state))
            return self.state, E_loc_i, d_ln_psi_i
        else:
            return self.state

    def clear_state(self):
        self.state = torch.rand(3, self.n_walkers)


class TrialPsi(nn.Module):
    def __init__(self, r_min: Position, r_max: Position, num_walkers: int):
        super(TrialPsi, self).__init__()
        self.theta = torch.rand(1) * 0.01
        self.mc = MarkovChain(self, r_min, r_max, num_walkers)

    def forward(self, x: Position) -> HilbertVector:
        return torch.exp(-self.theta[0] * torch.sum(x**2, axis=0))  # type: ignore

    def monte_carlo_sample(
        self,
        num_warmup: int,
        num_steps: int,
        step_size: float,
    ):
        self.mc.clear_state()

        E_loc = torch.tensor(0.0)
        d_ln_psi = torch.tensor(0.0)
        E_loc_d_ln_psi = torch.tensor(0.0)

        for i in range(num_warmup):
            self.mc.metropolis(step_size, calc_grad_terms=False)

        for i in range(num_steps):
            state_i, E_loc_i, d_ln_psi_i = self.mc.metropolis(
                step_size, calc_grad_terms=True
            )

            E_loc_d_ln_psi_i = E_loc_i * d_ln_psi_i

            E_loc += E_loc_i / num_steps
            d_ln_psi += d_ln_psi_i / num_steps
            E_loc_d_ln_psi += E_loc_d_ln_psi_i / num_steps

        # Everything needed to calculate the gradient
        return E_loc, d_ln_psi, E_loc_d_ln_psi


class TrialPsiOptimizer:
    def __init__(self, psi: TrialPsi, lr: float):
        self.psi = psi
        self.lr = lr

    def optimize(
        self, num_opt_iters: int, num_warmup: int, num_steps: int, step_size: float
    ):
        for i in (pbar := tqdm(range(num_opt_iters))):
            E_loc, d_ln_psi, E_loc_d_ln_psi = self.psi.monte_carlo_sample(
                num_warmup, num_steps, step_size
            )

            grad = 2 * (E_loc_d_ln_psi - E_loc * d_ln_psi)

            self.psi.theta -= self.lr * grad

            pbar.set_description(
                f"Energy: {E_loc.item():.4f} - Theta: {self.psi.theta.item():.4f} - Grad: {grad.item():.4f}"
            )


def show_density(
    psi: Psi,
    warmup_steps: int,
    steps: int,
    step_size: float,
    warmup_step_size: float,
    num_walkers: int,
    r_min: Position,
    r_max: Position,
    bins: int = 50,
):
    mc = MarkovChain(psi, r_min, r_max, num_walkers)

    # These are computation chains
    warmup_coords = torch.zeros(3, mc.n_walkers, warmup_steps)
    coords = torch.zeros(3, mc.n_walkers, steps)

    print("Running warmup steps")
    for i in tqdm(range(warmup_steps)):
        warmup_coords[:, :, i] = mc.metropolis(warmup_step_size)

    print("Running steps")
    for i in tqdm(range(steps)):
        coords[:, :, i] = mc.metropolis(step_size)

    if steps * num_walkers < 10000:
        fig = go.Figure()

        for i in range(steps):
            fig.add_trace(
                go.Scatter3d(
                    x=coords[0, :, i].tolist(),
                    y=coords[1, :, i].tolist(),
                    z=coords[2, :, i].tolist(),
                    mode="markers",
                    marker=dict(
                        size=1,
                        color=[i] * mc.n_walkers,
                        colorscale="Plasma",
                        colorbar=dict(title="Iteration"),
                        cmin=0,
                        cmax=steps,
                    ),
                    opacity=1,
                )
            )

        fig.update_layout(scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))
        fig.show()

    # Extract the x and y coordinates from the coords tensor
    x_coords = coords[0, :, :].flatten().tolist()
    y_coords = coords[1, :, :].flatten().tolist()

    # Create a 2D histogram
    plt.hist2d(x_coords, y_coords, bins=bins, cmap="plasma")

    # Set labels for the axes
    plt.xlabel("X")
    plt.ylabel("Y")

    # Add a colorbar
    plt.colorbar()

    # Show the plot
    plt.show()
