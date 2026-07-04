<<<<<<< HEAD
from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg


def server_fn(context: Context):
    num_rounds = int(context.run_config.get("num-server-rounds", 3))

    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=10,
        min_evaluate_clients=10,
        min_available_clients=10,
    )

    # ADDED: control_loop_timeout sets how long (in seconds) the server will wait for client responses.
    # We set it to 120.0 seconds to give PyTorch ample time to finish its slow operator registrations.
    config = ServerConfig(num_rounds=num_rounds, control_loop_timeout=120.0)
    
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
=======
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
>>>>>>> origin/dev
