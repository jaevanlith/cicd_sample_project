from utils import IO


def predict(cfg, model):
    '''
    Method to predict on unlabeled data and write to file.
    '''
    # Load data
    X, _ = IO.read_csv_to_numpy(cfg.data_dir + '/data_unlabeled.csv', labeled=False)

    # Predict
    y_pred = model.predict(X)

    # Write to file
    IO.write_to_csv(X, y_pred, 'predictions.csv')

