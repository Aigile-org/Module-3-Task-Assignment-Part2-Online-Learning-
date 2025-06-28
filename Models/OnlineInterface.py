
from river import feature_extraction, compose, drift
from running_models import RollingAccuracyCalc, RollingF1Calc, RollingPrecisionCalc, RollingRecallCalc
class OnlineInterface:
    """
    A blueprint for an online machine learning model for issue assignment.
    
    This class provides the core framework for:
    1. Feature extraction using TF-IDF.
    2. Tracking progressive accuracy.
    3. Detecting concept drift with ADWIN.z
    """
    def __init__(self, drift_delta=None):
        

            # If "all" is specified, we use our standard set of features.
        feature_names = ["summary", "description", "labels", "components_name", "priority_name", "issue_type_name"]


        # --- Part B: Set up the Drift Detector ---
        # The drift detector is optional. We only create it if a 'delta' value is given.
        if drift_delta is not None:
            self.drift_detector = drift.ADWIN(delta=drift_delta)
        else:
            self.drift_detector = None
        
        # --- Part C: Create the Feature Engineering Pipeline ---
        # This is the core of our feature preparation.
        # We create a list where each item is a TF-IDF transformer for one feature.
        # The `on=name` tells each transformer which key to look for in the input dictionary.
        list_of_transformers = []
        for name in feature_names:
            list_of_transformers.append(feature_extraction.TFIDF(on=name))
            
        # `Tranformer_Union_CalcformerUnion` combines the outputs of all individual transformers
        # into a single, large feature vector. The '*' unpacks our list into arguments.
        self.feature_pipeline = compose.TransformerUnion(*list_of_transformers)

        # --- Part D: Initialize containers for results ---
        self.accuracies_over_time = []  # To store accuracy at each step
        self.precisions_over_time = []
        self.recalls_over_time = []
        self.f1_scores_over_time = []
        self.detected_drifts_at = []    # To store the index 'i' where drift occurs
        # Initialize metric trackers
        self.running_metrics = {
            'accuracy': RollingAccuracyCalc(),
            'precision': RollingPrecisionCalc(),
            'recall': RollingRecallCalc(),
            'f1': RollingF1Calc()
        }
        
        # This is a placeholder. The actual ML algorithm (e.g., Naive Bayes)
        # will be defined in the child classes that inherit from this one.
        self.ml_model = None 

    def _transform_features(self, single_issue_dict):
        """
        Processes a single issue's features using the TF-IDF pipeline.
        
        This method does two things:
        1. Updates the pipeline's internal state (e.g., vocabulary) with `learn_one`.
        2. Converts the raw feature dictionary into a numerical vector with `transform_one`.
        """
        # First, let the pipeline learn from the raw data.
        self.feature_pipeline.learn_one(single_issue_dict)
        
        # Then, get the transformed numerical vector.
        transformed_vector = self.feature_pipeline.transform_one(single_issue_dict)
        
        return transformed_vector

    def _predict_and_learn(self, feature_vector, true_label):
        """
        Performs the 'test-then-train' cycle on the machine learning model.
        """
        # 1. PREDICT (Test): Make a prediction using the current state of the model.
        predicted_label = self.ml_model.predict_one(feature_vector)
        
        # 2. LEARN (Train): Update the model with the correct answer.
        self.ml_model.learn_one(feature_vector, true_label)
        
        return predicted_label
    
    # def _update_accuracy(self, true_label, predicted_label):
    #     """Updates the rolling accuracy and stores its current value."""
    #     # Update our custom metric object with the latest result
    #     self.running_accuracy.update(true_label, predicted_label)
    #     # Append the new overall accuracy to our list for later plotting
    #     self.accuracies_over_time.append(self.running_accuracy.get())
        
    def _check_for_drift(self, issue_index):
        if self.drift_detector is not None:
            # A more robust way to use ADWIN is to feed it a stream of 0s (error) and 1s (correct)
            is_correct = 1 if self.running_metrics['accuracy'].get() > (self.accuracies_over_time[-2] if len(self.accuracies_over_time) > 1 else 0) else 0
            self.drift_detector.update(is_correct)
            if self.drift_detector.drift_detected:
                self.detected_drifts_at.append(issue_index)

    def process_one_issue(self, issue_index, issue_features, issue_label):
        """
        The main public method to process a single issue through the entire pipeline.
        """
        # Step 1: Convert raw features into a numerical vector.
        feature_vector = self._transform_features(issue_features)
        
        # Step 2: Get a prediction and train the model.
        prediction = self._predict_and_learn(feature_vector, issue_label)
        
        # Step 3: Update our performance metrics.
        self._update_metrics(issue_label, prediction)
        self._check_for_drift(issue_index)
        
        return prediction

    def _update_metrics(self, true_label, predicted_label):
            """Updates all metrics and stores their current values."""
            # Update all metric objects
            for metric in self.running_metrics.values():
                metric.update(true_label, predicted_label)
            
            # Append the new metrics to our lists
            self.accuracies_over_time.append(self.running_metrics['accuracy'].get())
            self.precisions_over_time.append(self.running_metrics['precision'].get())
            self.recalls_over_time.append(self.running_metrics['recall'].get())
            self.f1_scores_over_time.append(self.running_metrics['f1'].get())

    def get_results(self):
        """Returns the final results of the simulation run."""
        return {
            "accuracies": self.accuracies_over_time,
            "precisions": self.precisions_over_time,
            "recalls": self.recalls_over_time,
            "f1_scores": self.f1_scores_over_time,
            "drifts": self.detected_drifts_at
        }
