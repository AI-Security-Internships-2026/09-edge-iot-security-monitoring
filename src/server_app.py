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