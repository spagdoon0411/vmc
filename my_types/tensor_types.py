from jaxtyping import Array, Float, Integer, Complex
from typing import Callable
import torch

type HilbertVector = Complex[torch.Tensor, "num_walkers"] | Float[
    torch.Tensor, "num_walkers"
]
type Psi = Callable[
    [Float[torch.Tensor, "3 num_walkers"]],
    Complex[torch.Tensor, "num_walkers"] | Float[torch.Tensor, "num_walkers"],
]
type Position = Float[torch.Tensor, "3 #num_walkers"]
type RealPositionFunction = Callable[[Position], Float[torch.Tensor, "num_walkers"]]
type ComplexPositionFunction = Callable[
    [Position], Complex[torch.Tensor, "num_walkers"]
]
type PositionFunctionBuffer = Complex[
    torch.Tensor, "n_pos_functions, n_walkers"
] | Float[torch.Tensor, "n_pos_functions, n_walkers"]
