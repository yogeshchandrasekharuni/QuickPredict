from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix, precision_recall_fscore_support

class Metrics:
    def __init__(self, y_test, predictions):
        self.y_test = y_test
        self.predictions = predictions

    def get_accuracy(self):
        return accuracy_score(self.y_test, self.predictions)

    def get_roc(self):
        return roc_curve(self.y_test, self.predictions)

    def get_confusion_matrix(self):
        return confusion_matrix(self.y_test, self.predictions)

    def get_prec_recall_f1(self):
        return precision_recall_fscore_support(self.y_test, self.predictions)

