from sklearn.preprocessing import StandardScaler, Normalizer, LabelEncoder

class Preprocess:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def normalize_features(self):
        normalizer = Normalizer()
        self.X_train = normalizer.fit_transform(self.X_train)
        self.X_test = normalizer.transform(self.X_test)
        return self.X_train, self.X_test

    def standardize_features(self):
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        return self.X_train, self.X_test

    def labelencode_responses(self):
        encoder = LabelEncoder()
        self.y_train = encoder.fit_transform(self.y_train)
        self.y_test = encoder.transform(self.y_test)
        return self.y_train, self.y_test