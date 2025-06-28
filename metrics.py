class RollingPrecisionCalc:
    def __init__(self):
        self.true_positives = 0
        self.false_positives = 0
    
    def update(self, y_true, y_pred):
        if y_pred == y_true:
            self.true_positives += 1
        elif y_pred != y_true:
            self.false_positives += 1
    
    def get(self):
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator > 0 else 0.0

class RollingRecallCalc:
    def __init__(self):
        self.true_positives = 0
        self.false_negatives = 0
    
    def update(self, y_true, y_pred):
        if y_pred == y_true:
            self.true_positives += 1
        elif y_pred != y_true and y_pred != "correct_prediction":
            self.false_negatives += 1
    
    def get(self):
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator > 0 else 0.0

class RollingF1Calc:
    def __init__(self):
        self.precision_calc = RollingPrecisionCalc()
        self.recall_calc = RollingRecallCalc()
    
    def update(self, y_true, y_pred):
        self.precision_calc.update(y_true, y_pred)
        self.recall_calc.update(y_true, y_pred)
    
    def get(self):
        precision = self.precision_calc.get()
        recall = self.recall_calc.get()
        denominator = precision + recall
        return 2 * (precision * recall) / denominator if denominator > 0 else 0.0

class RollingAccuracyCalc:
    """
    Custom implementation of accuracy metric for online learning.
    """
    def __init__(self):
        self.correct = 0
        self.total = 0
    
    def update(self, y_true, y_pred):
        """Update the accuracy metric with a new prediction."""
        self.total += 1
        if y_true == y_pred:
            self.correct += 1
    
    def get(self):
        """Return the current accuracy."""
        return self.correct / self.total if self.total > 0 else 0.0
    
    def __repr__(self):
        return f"Accuracy: {self.get():.4f}"
