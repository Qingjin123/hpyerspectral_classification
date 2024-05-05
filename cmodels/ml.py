from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelPropagation
from sklearn.model_selection import train_test_split


class HyperspectralPixelClassifier:

    def __init__(self):
        self.model = LabelPropagation()
        self.scaler = StandardScaler()
        self.y_true = None
        self.y_pred = None

    def getData(self, X, y_true):
        try:
            X_train, X_test, y_train, y_test = train_test_split(X,
                                                                y_true,
                                                                test_size=0.85)
            return X_train, X_test, y_train, y_test
        except ValueError:
            print("Error: Data shape is not compatible.")
            return None, None, None, None

    def preprocess(self, X):
        try:
            return self.scaler.fit_transform(X)
        except ValueError:
            print("Error: Failed to preprocess data.")
            return None

    def fit(self, X, y):
        try:
            X = self.preprocess(X)
            if X is not None:
                self.model.fit(X, y)
        except ValueError:
            print("Error: Failed to fit model.")

    def predict(self, X):
        try:
            X = self.preprocess(X)
            if X is not None:
                return self.model.predict(X)
            return None
        except ValueError:
            print("Error: Failed to predict.")

    def evaluate(self, X, y_true):
        try:
            y_pred = self.predict(X)
            if y_pred is not None:
                self.y_pred = y_pred
                self.y_true = y_true
        except ValueError:
            print("Error: Failed to evaluate.")

    def run(self, X, y_true):
        try:
            X_train, X_test, y_train, y_test = self.getData(X, y_true)
            if X_train is not None:
                self.fit(X_train, y_train)
                self.evaluate(X_test, y_test)
        except Exception as e:
            print(f"Error: {str(e)}")
