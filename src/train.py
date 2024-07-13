from models.linear_regression import LinearRegression
from sklearn.model_selection import train_test_split
import utils.IO as IO


def train(cfg, save_model=True):
    """
    Method to train a model.

    @param cfg: Hydra configuration object

    @return train_score: float
    @return test_score: float
    """

    # Load data
    X, y = IO.read_csv_to_numpy(cfg.data_dir + "/data_labeled.csv", labeled=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Initialize model
    if cfg.model == "LinearRegression":
        model = LinearRegression()
    else:
        raise ValueError(f"Invalid model: {cfg.model}")

    # Train model
    for i in range(cfg.epochs):
        model.fit(X_train, y_train)

    # Evaluate model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    # Save model
    if save_model:
        IO.save_model(model, cfg)

    return train_score, test_score, model
