from atoms.hydrogen_psi import TrialPsi
from tqdm import tqdm
import torch


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

            grad = torch.real(
                (E_loc_d_ln_psi + E_loc_d_ln_psi.conj())
                - (E_loc) * (d_ln_psi + d_ln_psi.conj())
            )

            self.psi.theta -= self.lr * grad

            torch.set_printoptions(precision=4)

            pbar.set_description(
                f"Energy: {E_loc} - Theta: {self.psi.theta} - Grad: {grad}"
            )

        torch.set_printoptions(profile="default")
