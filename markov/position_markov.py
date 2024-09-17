from my_types.tensor_types import Position, RealPositionFunction, Psi
import torch


class MarkovChain:
    def __init__(
        self,
        psi: Psi,
        r_min: Position,
        r_max: Position,
        n_walkers: int,
        E_loc: RealPositionFunction,
        d_ln_psi: RealPositionFunction,
    ):
        self.state: Position = torch.rand(3, n_walkers)

        # constants
        self.psi: Psi = psi
        self.n_walkers: int = n_walkers
        self.r_min: Position = r_min.reshape(3, 1)
        self.r_max: Position = r_max.reshape(3, 1)
        self.E_loc: RealPositionFunction = E_loc
        self.d_ln_psi: RealPositionFunction = d_ln_psi

    def make_step(self, step_size: float):
        step: Position = torch.randn(3, self.n_walkers)
        step = step * (self.r_max - self.r_min) * step_size
        return step

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
