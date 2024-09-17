from markov.position_markov import MarkovChain
from my_types.tensor_types import HilbertVector, Position
from jaxtyping import Float
import torch
from torch import nn


class TrialPsi(nn.Module):
    def __init__(self, r_min: Position, r_max: Position, num_walkers: int):
        super(TrialPsi, self).__init__()
        self.theta = torch.rand(1) * 0.01
        self.mc = MarkovChain(
            self, r_min, r_max, num_walkers, self.E_loc, self.d_ln_psi
        )

    def E_loc(self, position_samples: Position) -> Float[torch.Tensor, "num_walkers"]:
        alpha = self.theta[0]
        r = torch.norm(position_samples, dim=0)
        return -0.5 * alpha * (alpha - 2 / r) - 1 / r

    def d_ln_psi(
        self,
        position_samples: Position,
    ) -> Float[torch.Tensor, "num_walkers"]:
        alpha = self.theta[0]

        r = torch.norm(position_samples, dim=0)

        return -(r**2)

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
