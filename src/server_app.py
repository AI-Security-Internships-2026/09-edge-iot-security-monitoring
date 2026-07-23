import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from task import MLP

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    num_rounds        = context.run_config.get("num-server-rounds", 3)
    fraction_evaluate = context.run_config.get("fraction-evaluate", 1.0)
    lr                = context.run_config.get("learning-rate", 0.001)

    global_model = MLP()
    initial_arrays = ArrayRecord(global_model.state_dict())

    strategy = FedAvg(fraction_evaluate=fraction_evaluate)

    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    torch.save(result.arrays.to_torch_state_dict(), "final_model.pt")
