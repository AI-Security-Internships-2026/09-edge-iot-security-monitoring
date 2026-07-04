from sklearn.neural_network import MLPClassifier
import numpy as np

def get_model(num_features: int = 61, num_classes: int = 15):
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        warm_start=True
    )
    X_dummy = np.zeros((num_classes, num_features))
    y_dummy = np.arange(num_classes)
    all_classes = np.arange(num_classes)
    model.partial_fit(X_dummy, y_dummy, classes=all_classes)
    return model

def get_model_parameters(model):
    """Extracts all weights and biases as a list of NumPy arrays for Flower."""
    return model.coefs_ + model.intercepts_

def set_model_parameters(model, parameters):
    """Applies aggregated global weights back into the Scikit-Learn architecture."""
    if not parameters or len(parameters) == 0:
        return model
    
    n_layers = len(model.hidden_layer_sizes) + 1
    model.coefs_ = [np.array(p) for p in parameters[:n_layers]]
    model.intercepts_ = [np.array(p) for p in parameters[n_layers:]]
    return model

def train(model, X_train, y_train, epochs=2):
    all_classes = np.arange(15)
    for _ in range(epochs):
        model.partial_fit(X_train, y_train, classes=all_classes)
    return model

def test(model, X_test, y_test):
    """Evaluates accuracy and cross-entropy loss metrics."""
    accuracy = model.score(X_test, y_test)
    try:
        prob = model.predict_proba(X_test)
        prob = np.clip(prob, 1e-15, 1 - 1e-15)
        one_hot = np.eye(prob.shape[1])[y_test.astype(int)]
        loss = -np.mean(np.sum(one_hot * np.log(prob), axis=1))
    except Exception:
        loss = 0.0
    return float(loss), float(accuracy)