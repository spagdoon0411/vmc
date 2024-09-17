from markov.index_markov import MarkovChain
from my_types.tensor_types import StateTensor, ComplexFloat
import torch
from torch import nn


class TrialPsi(nn.Module):
    def __init__(self, t: float, num_sites: int, num_walkers: int):
        super(TrialPsi, self).__init__()
        self.theta = torch.rand(3) * 0.01
        self.mc = MarkovChain(self, num_sites, num_walkers, self.E_loc, self.d_ln_psi)
        self.num_sites = num_sites
        self.t = t

    def forward(self, r: StateTensor) -> ComplexFloat:
        return torch.exp(
            1j * self.theta[0] * r - self.theta[1] * (r - self.theta[2]) ** 2
        )

    def E_loc(self, r: StateTensor) -> ComplexFloat:
        idx_l = r - 1 % self.num_sites
        idx_r = r + 1 % self.num_sites
        psi_i = self.forward(r)
        psi_l = self.forward(idx_l)
        psi_r = self.forward(idx_r)
        return -(self.t / psi_i) * (psi_l + psi_r)

    def d_ln_psi(self, r: StateTensor) -> ComplexFloat:
        # TODO: negatives
        return torch.stack(
            [
                1j * r,
                (r - self.theta[2]) ** 2,
                2 * self.theta[1] * (r - self.theta[2]),
            ]
        )

    def monte_carlo_sample(
        self,
        num_warmup: int,
        num_steps: int,
        s_dev: float,
    ):
        self.mc.clear_state()

        E_loc = torch.tensor(0.0).to(torch.complex64)
        d_ln_psi = torch.tensor(0.0).to(torch.complex64)
        E_loc_d_ln_psi = torch.tensor(0.0).to(torch.complex64)

        for i in range(num_warmup):
            self.mc.metropolis(s_dev, calc_grad_terms=False)

        for i in range(num_steps):
            state_i, E_loc_i, d_ln_psi_i = self.mc.metropolis(
                s_dev, calc_grad_terms=True
            )

            E_loc_d_ln_psi_i = E_loc_i * d_ln_psi_i.conj()

            E_loc += E_loc_i / num_steps
            d_ln_psi += d_ln_psi_i / num_steps
            E_loc_d_ln_psi += E_loc_d_ln_psi_i / num_steps

        # Everything needed to calculate the gradient
        return E_loc, d_ln_psi, E_loc_d_ln_psi
