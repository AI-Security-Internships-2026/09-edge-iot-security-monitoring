import numpy as np
from client_app import FlowerClient, get_model_parameters, set_model_parameters
from task import get_model
from server_app import server_fn
from flwr.common import Context

NUM_CLIENTS = 10
NUM_ROUNDS = 3

def main():
    # Create clients
    clients = []
    for i in range(NUM_CLIENTS):
        X_train = np.random.randn(100, 61)
        y_train = np.random.randint(0, 15, size=100)
        X_test = np.random.randn(20, 61)
        y_test = np.random.randint(0, 15, size=20)
        clients.append(FlowerClient(X_train, y_train, X_test, y_test))

    # Get initial parameters from first client
    global_params = clients[0].get_parameters(config={})

    # FL rounds
    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"\n[ROUND {round_num}/{NUM_ROUNDS}]")

        # Each client trains
        all_params = []
        all_sizes = []
        for i, client in enumerate(clients):
            params, size, _ = client.fit(global_params, config={"lr": 0.001})
            all_params.append((params, size))
            all_sizes.append(size)

        # FedAvg aggregation
        total = sum(all_sizes)
        global_params = [
            sum(p[j] * (s / total) for p, s in all_params)
            for j in range(len(global_params))
        ]

        # Evaluate
        losses, accs = [], []
        for client in clients:
            loss, size, metrics = client.evaluate(global_params, config={})
            losses.append(loss)
            accs.append(metrics["accuracy"])

        print(f"Loss: {np.mean(losses):.4f} | Accuracy: {np.mean(accs):.4f}")

    print("\nTraining complete.")

if __name__ == "__main__":
    main()