import pandas as pd
import pickle
from pathlib import Path


def read_csv_to_numpy(file_path, labeled=True):
    """
    Method to read a csv file and return the data as numpy arrays.

    @param file_path: str
    @param labeled: bool

    @return X: np.array
    @return y: np.array
    """
    # Read file
    assert file_path.endswith(".csv"), "file_path must end with .csv"
    df = pd.read_csv(file_path)

    # Extract data
    assert "x1" in df.columns, "x1 column not found"
    assert "x2" in df.columns, "x2 column not found"
    assert "x3" in df.columns, "x3 column not found"
    X = df[["x1", "x2", "x3"]].to_numpy()

    # Extract labels
    y = None
    if labeled:
        assert "y" in df.columns, "y column not found"
        y = df["y"].to_numpy()

    return X, y


def write_to_csv(X, y, filename):
    """
    Method to write a dataframe to a csv file.

    @param X: np.array
    @param y: np.array
    @param filename: str
    """
    # Create dataframe
    df = pd.DataFrame(X, columns=["x1", "x2", "x3"])
    if y is not None:
        df["y"] = y

    # Write to file
    assert filename.endswith(".csv"), "filename must end with .csv"
    df.to_csv(filename)


def save_model(obj, cfg):
    """
    Method to write a model to a pickle file.

    @param obj: object
    @param cfg: Hydra configuration object
    """
    directory_path = Path(cfg.model_dir)
    directory_path.mkdir(parents=True, exist_ok=True)

    try:
        with open(cfg.model_dir + "/" + cfg.model + ".pkl", "wb") as f:
            pickle.dump(obj, f)
    except Exception as e:
        raise Exception(f"Error saving model: {e}")


def load_model(filename):
    """
    Method to read a model from a pickle file.

    @param filename: str

    @return model: object
    """
    assert filename.endswith(".pkl"), "filename must end with .pkl"

    try:
        with open(filename, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        raise Exception(f"Error loading model: {e}")

    return model
