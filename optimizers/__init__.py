from tqdm import tqdm
from atoms.hydrogen_psi import TrialPsi


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
