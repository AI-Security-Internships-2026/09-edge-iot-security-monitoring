import numpy as np
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from task import get_model, get_model_parameters, set_model_parameters, train, test


class FlowerClient(NumPyClient):
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = get_model(num_features=X_train.shape[1])

    def get_parameters(self, config):
        return get_model_parameters(self.model)

    def fit(self, parameters, config):
        set_model_parameters(self.model, parameters)
        train(self.model, self.X_train, self.y_train, epochs=2)
        return get_model_parameters(self.model), len(self.X_train), {}

    def evaluate(self, parameters, config):
        set_model_parameters(self.model, parameters)
        loss, accuracy = test(self.model, self.X_test, self.y_test)
        return float(loss), len(self.X_test), {"accuracy": float(accuracy)}


def client_fn(context: Context):
    X_train = np.random.randn(100, 61)
    y_train = np.random.randint(0, 15, size=100)
    X_test = np.random.randn(20, 61)
    y_test = np.random.randint(0, 15, size=20)
    return FlowerClient(X_train, y_train, X_test, y_test).to_client()


app = ClientApp(client_fn=client_fn)