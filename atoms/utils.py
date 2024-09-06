from atoms.hydrogen_psi import TrialPsi
from markov.position_markov import MarkovChain
from my_types.tensor_types import Position
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import torch


def show_density(
    psi: TrialPsi,
    warmup_steps: int,
    steps: int,
    step_size: float,
    warmup_step_size: float,
    r_min: Position,
    r_max: Position,
    bins: int = 50,
    show_paths: bool = False,
):
    mc = psi.mc  # MarkovChain(psi, r_min, r_max, num_walkers)
    mc.clear_state()

    warmup_coords = torch.zeros(3, mc.n_walkers, warmup_steps)
    coords = torch.zeros(3, mc.n_walkers, steps)

    print("Running warmup steps")
    for i in tqdm(range(warmup_steps)):
        warmup_coords[:, :, i] = mc.metropolis(warmup_step_size)  # type: ignore

    print("Running steps")
    for i in tqdm(range(steps)):
        coords[:, :, i] = mc.metropolis(step_size)  # type: ignore

    if show_paths:
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
