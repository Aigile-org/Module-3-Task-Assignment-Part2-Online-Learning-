# ==============================================================================
# Part 0: Imports and Configuration
# All necessary libraries are imported at the top.
# ==============================================================================

import pandas as pd
from collections import defaultdict
from itertools import repeat
import random
# River is the core online machine learning library
from river import feature_extraction, compose, naive_bayes, multiclass, ensemble, drift, metrics,  linear_model ,tree, optim
from sklearn.linear_model import PassiveAggressiveClassifier



# ==============================================================================
# Part 1: Online Model Definitions (from online_models.py)
# This section contains the class definitions for the online learning framework.
# ==============================================================================

class MyOnlineModel:
    """
    A blueprint for an o
    nline machine learning model for issue assignment.
    
    This class provides the core framework for:
    1. Feature extraction using TF-IDF.
    2. Tracking progressive accuracy.
    3. Detecting concept drift with ADWIN.
    """
    def __init__(self, variables_to_use="all", drift_delta=None):
        
        # --- Part A: Define which features to use ---
        if variables_to_use == "all":
            # If "all" is specified, we use our standard set of features.
            feature_names = ["summary", "description", "labels", "components_name", "priority_name", "issue_type_name"]
        else:
            # Otherwise, we use the specific list of features provided.
            feature_names = variables_to_use

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
            
        # `TransformerUnion` combines the outputs of all individual transformers
        # into a single, large feature vector. The '*' unpacks our list into arguments.
        self.feature_pipeline = compose.TransformerUnion(*list_of_transformers)

        # --- Part D: Initialize containers for results ---
        self.accuracies_over_time = []  # To store accuracy at each step
        self.detected_drifts_at = []    # To store the index 'i' where drift occurs
        self.running_accuracy = metrics.Accuracy() # The river object to calculate rolling accuracy
        
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
    #TODO:: Know How Its calculated ( Progressive Accuracy ) 
    def _update_accuracy(self, true_label, predicted_label):
        """Updates the rolling accuracy and stores its current value."""
        # Update the river metric object with the latest result
        self.running_accuracy.update(true_label, predicted_label)
        
        # Append the new overall accuracy to our list for later plotting
        self.accuracies_over_time.append(self.running_accuracy.get())
        
    #TODO:: Understand This
    def _check_for_drift(self, issue_index):
        if self.drift_detector is not None:
            # A more robust way to use ADWIN is to feed it a stream of 0s (error) and 1s (correct)
            is_correct = 1 if self.running_accuracy.get() > (self.accuracies_over_time[-2] if len(self.accuracies_over_time) > 1 else 0) else 0
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
        self._update_accuracy(issue_label, prediction)
        self._check_for_drift(issue_index)
        
        return prediction

    def get_results(self):
        """Returns the final results of the simulation run."""
        return {
            "accuracies": self.accuracies_over_time, 
            "drifts": self.detected_drifts_at
        }

# This class inherits from the MyOnlineModel we just wrote.
# This means it gets all its methods (_transform_features, process_one_issue, etc.) automatically.
class MyNaiveBayesModel(MyOnlineModel):
    """
    A simple online classifier using the Multinomial Naive Bayes algorithm.
    It relies entirely on the parent MyOnlineModel for its functionality.
    """
    def __init__(self, alpha=0.1, variables_to_use="all"):
        # --- Step A: Initialize the parent class ---
        # `super()` refers to the parent class (MyOnlineModel).
        # We call its __init__ method to set up the feature pipeline,
        # the accuracy trackers, and everything else.
        super().__init__(variables_to_use=variables_to_use)
        
        # --- Step B: Define the specific ML model for this class ---
        # We now assign a specific algorithm to the `self.ml_model` placeholder
        # that was left empty in the parent class.
        self.ml_model = naive_bayes.MultinomialNB(alpha=alpha)
        
        # --- Step C: Give our model a name for reports and file paths ---
        self.name = "My_Naive_Bayes"

class MyNaiveBayesWithADWIN(MyOnlineModel):
    """
    An online Naive Bayes classifier that resets itself when concept drift is detected.
    """
    def __init__(self, alpha=0.1, delta=0.15, variables_to_use="all"):
        # --- Step A: Initialize the parent class, but this time passing the delta ---
        # Because we provide a 'drift_delta', the parent's __init__ method
        # will now create an instance of the ADWIN detector.
        super().__init__(variables_to_use=variables_to_use, drift_delta=delta)
        
        # --- Step B: Define the specific ML model (same as before) ---
        self.ml_model = naive_bayes.MultinomialNB(alpha=alpha)
        
        # --- Step C: Give it a different name ---
        self.name = "My_Naive_Bayes_with_ADWIN"
        
    def process_one_issue(self, issue_index, issue_features, issue_label):
        """
        Overrides the parent method to add custom drift-handling logic.
        """
        # --- Step 1: Let the parent do its regular job first ---
        # We still want to do everything the parent does (transform, predict, learn, update accuracy).
        # Calling `super().process_one_issue(...)` runs the original method from MyOnlineModel.
        prediction = super().process_one_issue(issue_index, issue_features, issue_label)
        # --- Step 2: Add our own new logic AFTER the parent is done ---
        # The parent method has already checked for drift and updated the `detected_drifts_at` list.
        # We can check that list to see if a drift just occurred.
        # A simple way to check if drift happened *on this step* is to see if the last
        # item in the drift list matches the current issue index.
        if self.detected_drifts_at and self.detected_drifts_at[-1] == issue_index:
            # `.clone()` creates a fresh, untrained copy of our model
            # with the same initial hyperparameters (like `alpha`).
            self.ml_model = self.ml_model.clone()

        return prediction


# This class inherits all the base functionality from MyOnlineModel
class MyEnhancedModel(MyOnlineModel):
    """
    An advanced online classifier using an AdaBoost ensemble.
    
    It features two major enhancements:
    1. A custom prediction method that weights recently active developers more heavily.
    2. A smart drift-handling mechanism that only resets the worst-performing base models.
    """
    def __init__(self, alpha=0.05, delta=0.05, n_models=10, variables_to_use="all"):
        # --- Step A: Initialize the parent class ---
        # We pass the drift_delta to the parent to ensure the ADWIN detector is created.
        super().__init__(variables_to_use=variables_to_use, drift_delta=delta)
        
        # --- Step B: Define the specific ML model for this class ---
        # This is a two-part process for AdaBoost.
        
        # 1. Define the "weak learner" or base model. We'll use Naive Bayes.
        #    OneVsRestClassifier helps Naive Bayes handle multiple assignees.
        base_learner = multiclass.OneVsRestClassifier(naive_bayes.MultinomialNB(alpha=alpha))
        
        # 2. Create the AdaBoost ensemble, which is a "team" of these base learners.
        self.ml_model = ensemble.AdaBoostClassifier(model=base_learner, n_models=n_models, seed=42)
        
        # --- Step C: Initialize properties specific to this enhanced model ---
        
        # We need to track the accuracy of each individual model in the ensemble
        # to know which ones to reset when drift occurs.
        self.n_models = n_models
        self.base_model_accuracies = [metrics.Accuracy() for _ in range(self.n_models)]
        
        # These are for our recency heuristic.
        self.last_100_assignees = []
        self.recency_weights = defaultdict(lambda: 1.0) # Start with all weights at 1.0
        
        # --- Step D: Give our model a unique name ---
        self.name = "My_Enhanced_Model"

    def _predict_and_learn(self, feature_vector, true_label):
        """
        Overrides the parent method to include the developer recency heuristic
        before making a prediction.
        """
        # --- Part 1: Update Recency Window and Weights ---
        # Add the latest true assignee to our recency list.
        self.last_100_assignees.append(true_label)
        # Keep the list at a maximum size of 100.
        if len(self.last_100_assignees) > 100:
            self.last_100_assignees.pop(0) # Remove the oldest entry
        
        # Now, calculate the weights. Only developers seen in the last 100 issues
        # are considered "active".
        active_developers = set(self.last_100_assignees)
        self.recency_weights.clear() # Clear old weights
        for dev in active_developers:
            self.recency_weights[dev] = 1.0
        # Any developer NOT in the set will have a default weight of 0.0
        # because of how defaultdict works with a default factory of float (which is 0.0)
        # To be explicit, let's redefine the defaultdict
        self.recency_weights = defaultdict(lambda: 0.0, self.recency_weights)


        # --- Part 2: Make a Weighted Prediction ---
        # Get the probability scores from the AdaBoost model for each possible assignee.
        probabilities = self.ml_model.predict_proba_one(feature_vector)
        
        predicted_label = None
        if probabilities:
            # Apply our recency weights to the probabilities.
            weighted_probabilities = {}
            for dev, proba in probabilities.items():
                weighted_probabilities[dev] = proba * self.recency_weights[dev]
            
            # The final prediction is the developer with the highest weighted score.
            predicted_label = max(weighted_probabilities, key=weighted_probabilities.get)

        # --- Part 3: Train the Model (the "Learn" step) ---
        # This is crucial: we train the model AFTER making our prediction.
        self.ml_model.learn_one(feature_vector, true_label)
        
        return predicted_label

    def process_one_issue(self, issue_index, issue_features, issue_label):
        """
        Orchestrates the full process for one issue and adds advanced drift handling.
        """
        # --- Step A: Standard Feature Transformation and Prediction/Learning ---
        # We can reuse the parent's methods for these parts.
        feature_vector = self._transform_features(issue_features)
        prediction = self._predict_and_learn(feature_vector, issue_label) # Calls our OWN _predict_and_learn
        self._update_accuracy(issue_label, prediction) # Updates the main accuracy

        # --- Step B: Update Accuracies of Individual Base Models ---
        # To know which models are weak, we must track each one's performance.
        for i, model in enumerate(self.ml_model.models):
            base_pred = model.predict_one(feature_vector)
            if base_pred is not None:
                self.base_model_accuracies[i].update(issue_label, base_pred)

        # --- Step C: Check for Drift and Adapt Intelligently ---
        if self.drift_detector is not None:
            self.drift_detector.update(self.running_accuracy.get())
            
            if self.drift_detector.drift_detected:
                self.detected_drifts_at.append(issue_index)
                
                # The "Smart Reset" Logic:
                # 1. Get the current accuracies of all base models.
                model_accuracies = [(i, acc.get()) for i, acc in enumerate(self.base_model_accuracies)]
                
                # 2. Sort them from worst to best performance.
                model_accuracies.sort(key=lambda item: item[1])
                
                # 3. Identify the worst-performing half of the models.
                num_to_reset = self.n_models // 2
                indices_to_reset = [idx for idx, acc in model_accuracies[:num_to_reset]]
                
                # 4. Replace each weak model with a fresh, untrained clone.
                for idx in indices_to_reset:
                    self.ml_model.models[idx] = self.ml_model.model.clone()
                    # Also reset its performance history within the AdaBoost algorithm
                    self.ml_model.correct_weight[idx] = 0.0
                    self.ml_model.wrong_weight[idx] = 0.0
                    # And reset our own accuracy tracker for it.
                    self.base_model_accuracies[idx] = metrics.Accuracy()
                    
        return prediction
     
### ========================================================== ###
###           MODEL A: HOEFFDING ADAPTIVE TREE                 ###
### ========================================================== ###

class HoeffdingAdaptiveTreeModel(MyOnlineModel):
    """
    An online classifier using a Hoeffding Adaptive Tree (HAT).
    This model is designed for streaming data and has its own internal drift detection,
    making the parent's drift detector redundant.
    """
    def __init__(self, grace_period=150, delta_split=1e-5, delta_drift=0.01, **kwargs):
        # We don't pass drift_delta to the parent because this model handles drift internally.
        super().__init__(drift_delta=None, **kwargs) 
        
        self.name = "Hoeffding_Adaptive_Tree"
        self.ml_model = tree.HoeffdingAdaptiveTreeClassifier(
            grace_period=grace_period,
            delta=delta_split,
            drift_detector=drift.ADWIN(delta=delta_drift), # Internal drift detector
            seed=42
        )

    def _predict_and_learn(self, feature_vector, true_label):
        # This model provides probabilities, which is more robust
        probabilities = self.ml_model.predict_proba_one(feature_vector)
        predicted_label = max(probabilities, key=probabilities.get) if probabilities else None
        self.ml_model.learn_one(feature_vector, true_label)
        return predicted_label

    def process_one_issue(self, issue_index, issue_features, issue_label):
        """
        Simplified orchestration: We don't need the parent's drift check.
        """
        feature_vector = self._transform_features(issue_features)
        prediction = self._predict_and_learn(feature_vector, issue_label)
        self._update_accuracy(issue_label, prediction)
        # Note: No call to _check_for_drift()
        return prediction

### ========================================================== ###
###           MODEL B: LEVERAGING BAGGING CLASSIFIER           ###
### ========================================================== ###

class LeveragingBaggingModel(MyOnlineModel):
    """
    An online classifier using the Leveraging Bagging ensemble method.
    This is like an online Random Forest and handles drift internally.
    """
    def __init__(self, n_models=10, **kwargs):
        super().__init__(drift_delta=None, **kwargs)
        
        self.name = "Leveraging_Bagging"
        self.ml_model = ensemble.LeveragingBaggingClassifier(
            model=tree.HoeffdingTreeClassifier(grace_period=100), # Base learners
            n_models=n_models,
            seed=42
        )

    def _predict_and_learn(self, feature_vector, true_label):
        probabilities = self.ml_model.predict_proba_one(feature_vector)
        predicted_label = max(probabilities, key=probabilities.get) if probabilities else None
        self.ml_model.learn_one(feature_vector, true_label)
        return predicted_label

    def process_one_issue(self, issue_index, issue_features, issue_label):
        """
        Simplified orchestration: No need for the parent's drift check.
        """
        feature_vector = self._transform_features(issue_features)
        prediction = self._predict_and_learn(feature_vector, issue_label)
        self._update_accuracy(issue_label, prediction)
        return prediction

### ========================================================== ###
###           MODEL C: PASSIVE-AGGRESSIVE CLASSIFIER           ###
### ========================================================== ###

class PassiveAggressiveModel(MyOnlineModel):
    """
    An online classifier using a simple but effective Passive-Aggressive algorithm.
    It relies on the parent class for drift detection.
    """
    def __init__(self, drift_delta=0.05, **kwargs):
        # We DO pass drift_delta to the parent because this model needs it.
        super().__init__(drift_delta=drift_delta, **kwargs)
        
        self.name = "Passive_Aggressive"
        # PassiveAggressiveClassifier is a linear model, wrapped in OneVsRest
        # to handle the multi-class assignment problem.
        self.ml_model = multiclass.OneVsRestClassifier(
            classifier=linear_model.SoftmaxRegression(
                optimizer=optim.Adam(0.01),
                # loss=lo.Hinge()
            )
        )
        # Note: We don't need to override process_one_issue. The parent's default works perfectly.


### ========================================================== ###
###    YOUR ENHANCED MODEL WITH PROBABILISTIC RESET (SUPER)    ###
### ========================================================== ###

### ========================================================== ###
###    UPDATED MySuperEnhancedModel WITH DEPLOYMENT METHODS    ###
### ========================================================== ###

class MySuperEnhancedModel(MyOnlineModel):
    """
    A "super" enhanced version of the original AdaBoost model.
    It now includes methods for a real-world deployment scenario where
    prediction and learning are separate steps.
    """
    def __init__(self, alpha=0.1, delta=0.15, n_models=10, name_mapping=None,**kwargs):
        # This super().__init__() call correctly initializes the pipeline
        # from MyOnlineModel, which looks for 'summary', 'description', etc.
        super().__init__(drift_delta=delta, **kwargs)
        self.name_mapping = name_mapping or []

        
        self.name = "My_Super_Enhanced_Model"
        
        base_learner = multiclass.OneVsRestClassifier(naive_bayes.MultinomialNB(alpha=alpha))
        self.ml_model = ensemble.AdaBoostClassifier(model=base_learner, n_models=n_models, seed=42)
        
        self.n_models = n_models
        self.base_model_accuracies = [metrics.Accuracy() for _ in range(self.n_models)]
        self.recency_window = 100
        self.last_assignees = []
        
    # --- Existing methods for simulation ---
    # _predict_and_learn and process_one_issue remain unchanged
    # so your existing experiment runner still works.
    def _predict_and_learn(self, feature_vector, true_label):
        # (This method is used by the simulation runner)
        self.last_assignees.append(true_label)
        if len(self.last_assignees) > self.recency_window: self.last_assignees.pop(0)
        recency_weights = defaultdict(float)
        decay_factor = 0.98; current_weight = 1.0
        for dev in reversed(self.last_assignees):
            if dev not in recency_weights:
                recency_weights[dev] = current_weight
                current_weight *= decay_factor

        probabilities = self.ml_model.predict_proba_one(feature_vector)
        predicted_label = None
        if probabilities:
            weighted_probabilities = {dev: proba * recency_weights.get(dev, 0.0) for dev, proba in probabilities.items()}
            if any(weighted_probabilities.values()):
                predicted_label = max(weighted_probabilities, key=weighted_probabilities.get)
            else:
                predicted_label = max(probabilities, key=probabilities.get)
        self.ml_model.learn_one(feature_vector, true_label)
        return predicted_label

    def process_one_issue(self, issue_index, issue_features, issue_label):
        # (This method is used by the simulation runner)
        feature_vector = self._transform_features(issue_features)
        prediction = self._predict_and_learn(feature_vector, issue_label)
        self._update_accuracy(issue_label, prediction)
        for i, model in enumerate(self.ml_model.models):
            if (base_pred := model.predict_one(feature_vector)) is not None:
                self.base_model_accuracies[i].update(issue_label, base_pred)
        if self.drift_detector is not None:
            is_correct = 1 if prediction == issue_label else 0
            self.drift_detector.update(is_correct)
            if self.drift_detector.drift_detected:
                self.detected_drifts_at.append(issue_index)
                for i in range(self.n_models):
                    accuracy = self.base_model_accuracies[i].get()
                    if random.random() < (1.0 - accuracy):
                        self.ml_model.models[i] = self.ml_model.model.clone()
                        self.ml_model.correct_weight[i], self.ml_model.wrong_weight[i] = 0.0, 0.0
                        self.base_model_accuracies[i] = metrics.Accuracy()
        return prediction

    # --- NEW METHODS FOR DEPLOYMENT SIMULATION ---

    def predict_recommendations(self, issue_features: dict, top_n=3):
        """
        Step 1 of Deployment: Predicts assignees for a new issue with an unknown label.
        
        Returns:
            - A list of top_n recommended assignees (with actual names if mapping exists)
            - The generated feature_vector (to be used later in the learning step)
        """
        # First, transform the raw features
        feature_vector = self._transform_features(issue_features)
        
        # Get probabilities from the core model
        probabilities = self.ml_model.predict_proba_one(feature_vector)
        
        if not probabilities:
            return [], feature_vector

        # Apply recency weights
        recency_weights = defaultdict(float)
        decay_factor = 0.95
        current_weight = 1.0
        for dev in reversed(self.last_assignees):
            if dev not in recency_weights:
                recency_weights[dev] = current_weight
                current_weight *= decay_factor
        
        # Combine content-based and recency-based predictions
        weighted_scores = {}
        for dev, proba in probabilities.items():
            # Balance between content prediction (0.7) and recency (0.3)
            weighted_scores[dev] = (0.99 * proba) + (0.01 * recency_weights.get(dev, 0.0))
        
        # Sort by combined score
        sorted_recommendations = sorted(weighted_scores.items(), 
                                    key=lambda item: item[1], 
                                    reverse=True)
        
        # Convert numerical labels to names if mapping exists
        top_recommendations = []
        for dev_num, score in sorted_recommendations[:top_n]:
            # If we have a name mapping and the prediction is a number within range
            if hasattr(self, 'name_mapping') and self.name_mapping and isinstance(dev_num, int) and 0 <= dev_num < len(self.name_mapping):
                top_recommendations.append(self.name_mapping[dev_num])
            else:
                # Either no mapping or already a name (during training)
                top_recommendations.append(str(dev_num))
        
        return top_recommendations, feature_vector

    def learn_from_assignment(self, feature_vector: dict, true_label: str, issue_index: int):
        """
        Step 2 of Deployment: Learns from the assignment after the true label is known.
        This method contains the logic for updating all internal model states.
        """
        # --- Part 1: Update the core model ---
        self.ml_model.learn_one(feature_vector, true_label)
        
        # --- Part 2: Update recency heuristic list ---
        self.last_assignees.append(true_label)
        if len(self.last_assignees) > self.recency_window:
            self.last_assignees.pop(0)

        # --- Part 3: Update accuracy metrics ---
        # Generate a prediction to update accuracy metrics
        prediction = self._predict_for_accuracy(feature_vector)
        if prediction is not None:
            self._update_accuracy(true_label, prediction)
        
        # Update base model accuracies
        for i, model in enumerate(self.ml_model.models):
            if (base_pred := model.predict_one(feature_vector)) is not None:
                self.base_model_accuracies[i].update(true_label, base_pred)

        # Update and check the ADWIN drift detector
        if self.drift_detector is not None:
            is_correct = 1 if prediction == true_label else 0
            self.drift_detector.update(is_correct)
            
            if self.drift_detector.drift_detected:
                self.detected_drifts_at.append(issue_index)
                print(f"  -> Drift detected! Applying probabilistic reset...")
                for i in range(self.n_models):
                    accuracy = self.base_model_accuracies[i].get()
                    if random.random() < (1.0 - accuracy):
                        self.ml_model.models[i] = self.ml_model.model.clone()
                        self.ml_model.correct_weight[i], self.ml_model.wrong_weight[i] = 0.0, 0.0
                        self.base_model_accuracies[i] = metrics.Accuracy()

    def _predict_for_accuracy(self, feature_vector):
        """Helper method to generate predictions for accuracy tracking"""
        probabilities = self.ml_model.predict_proba_one(feature_vector)
        if probabilities:
            return max(probabilities, key=probabilities.get)
        return None
