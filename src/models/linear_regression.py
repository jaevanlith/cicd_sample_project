from sklearn.linear_model import LinearRegression

class MyLinearRegressionModel(LinearRegression):
    def __init__(self):
        super().__init__()