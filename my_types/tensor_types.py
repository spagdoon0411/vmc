from jaxtyping import Complex, Float, Int
from typing import Callable
import torch

ComplexFloat = Complex[torch.Tensor, "num_walkers"]
ComplexFloatGrad = Complex[torch.Tensor, "num_walkers dim_theta"]
ComplexFloatSingle = Complex[torch.Tensor, "1"]
ComplexFloatGradSingle = Complex[torch.Tensor, "dim_theta"]
StateTensor = Int[torch.Tensor, "num_walkers"]

Psi_Callable = Callable[[StateTensor], ComplexFloat]
E_loc_Callable = Callable[[StateTensor, ComplexFloat], ComplexFloatSingle]
d_ln_psi_Callable = Callable[[StateTensor, ComplexFloat], ComplexFloatGrad]
ProbDistTensor = Float[torch.Tensor, "num_walkers"]

Position = Float[torch.Tensor, "3 num_walkers"]
HilbertVector = (
    Complex[torch.Tensor, "num_walkers"] | Float[torch.Tensor, "num_walkers"]
)
RealPositionFunction = Callable[[Position], Float[torch.Tensor, "num_walkers"]]
Psi = Callable[[Position], HilbertVector]
