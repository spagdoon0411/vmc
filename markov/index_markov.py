from my_types.tensor_types import (
    E_loc_Callable,
    d_ln_psi_Callable,
    Psi_Callable,
    StateTensor,
)
import torch


class MarkovChain:
    def __init__(
        self,
        psi: Psi_Callable,
        n_sites: int,
        n_walkers: int,
        E_loc: E_loc_Callable,
        d_ln_psi: d_ln_psi_Callable,
    ):
        """
        Takes a Psi: StateTensor -> ComplexFloat, an E_loc: StateTensor -> ComplexFloat,
        and a d_ln_psi: StateTensor -> ComplexFloatGrad and initializes a Markov chain
        for an n-site chain with n walkers to average expected values over.
        """

        self.state: StateTensor = torch.randint(0, n_sites, (n_walkers,))

        # constants
        self.psi = psi
        self.n_walkers = n_walkers
        self.n_sites = n_sites
        self.E_loc = E_loc
        self.d_ln_psi = d_ln_psi

    def make_step(self, s_dev: int) -> StateTensor:
        """
        Returns proposed steps for each walker using a Gaussian distribution
        of steps with deviation `s_dev` and mean 0.
        """

        step = torch.normal(mean=0.0, std=s_dev, size=(self.n_walkers,))
        step = step.to(torch.int64)
        return step

    def metropolis(self, s_dev: float, calc_grad_terms: bool = False):
        """
        Perform a Metropolis-Hastings step for each walker in the Markov Chain.
        """
        proposed_step = self.make_step(s_dev)
        proposed_state = self.state + proposed_step

        prob_dist = self.psi(self.state)
        prob_dist = torch.real(prob_dist.conj() * prob_dist)

        proposed_prob_dist = self.psi(proposed_state)
        proposed_prob_dist = torch.real(proposed_prob_dist.conj() * proposed_prob_dist)

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
        """
        Resets the Markov chain state to a random state over a uniform
        distribution.
        """
        self.state = torch.randint(0, self.n_sites, (self.n_walkers,))
