# ==============================================================================
# Part 0: Imports and Configuration
# All necessary libraries are imported at the top.
# ==============================================================================
import os
import json
import pandas as pd
from collections import defaultdict, Counter
from itertools import repeat
import multiprocessing as mp
import random
import pickle 
import math
# River is the core online machine learning library
from river import feature_extraction, compose, naive_bayes, multiclass, ensemble, drift, metrics,  linear_model ,tree, optim
from parametars import TEST
# TEST = True
if not TEST:
    DATA_FOLDER = "data/"
else:
    DATA_FOLDER = "deploy and test\csvs"    
RESULTS_FOLDER = "results_new_1/"
MODEL_SAVE_FOLDER = "saved_models/"


# ==============================================================================
# Part 1: Helper Functions
# ==============================================================================

def my_preprocess_data(raw_data):
    """
    Prepares raw issue data for online machine learning with River.
    
    This function will:
    1. Sort the issues chronologically.
    2. Clean text fields.
    3. Separate features (X) and target (Y).
    4. Format X into a list of dictionaries.
    """    
    # Create a copy to avoid changing the original DataFrame
    data = raw_data.copy()
    
    # --- Step 1 & 2: Convert date strings to sortable integers ---
    data['created_date'] = pd.to_datetime(data['created_date'])
    data['sortable_date'] = data['created_date'].apply(lambda date_obj: date_obj.toordinal())
    
    # --- Step 3: Sort the entire dataset by date, oldest to newest ---
    data = data.sort_values(by='sortable_date', ascending=True)
    data = data.reset_index(drop=True)
    
    # --- Step 4: Clean data and separate features from the target ---
    feature_columns = ['summary', 'description', 'labels', 'components_name', 'priority_name', 'issue_type_name']
    
    for col in feature_columns:
        if col in data.columns:
            data[col] = data[col].fillna('')
    y_target = data['assignee']
    x_features_df = data[feature_columns]
    
    # --- Step 5: Convert the features DataFrame into a list of dictionaries for River ---
    x_features_stream = x_features_df.to_dict('records')

    return x_features_stream, y_target

# ==============================================================================
# Part 2: Online Model Definitions (from online_models.py)
# This section contains the class definitions for the online learning framework.
# ==============================================================================

class MyOnlineModel:
    """
    A blueprint for an online machine learning model for issue assignment.
    
    This class provides the core framework for:
    1. Feature extraction using TF-IDF.
    2. Tracking progressive accuracy.
    3. Detecting concept drift with ADWIN.
    """
    def __init__(self, drift_delta=None):
        
       
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
            list_of_transformers.append(TFIDF_Calc(on=name))
            
        # `Tranformer_Union_CalcformerUnion` combines the outputs of all individual transformers
        # into a single, large feature vector. The '*' unpacks our list into arguments.
        self.feature_pipeline = TransformUnion_calc(*list_of_transformers)

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
        elif y_pred != y_true and y_pred != "correct_prediction":  # Adjust based on your labels
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


class TFIDF_Calc:
    """Memory-efficient TF-IDF implementation with proper River compatibility."""
    
    def __init__(self, on: str):
        """
        Args:
            on (str): The key corresponding to the text field in the input dictionary.
        """
        self.on = on
        
        # State-tracking variables for online learning
        self.doc_count = 0
        self.doc_freqs = {}  # Stores document frequency for each term
        self.vocabulary = {} # Maps each term to a unique feature index
        self._next_vocab_idx = 0

    def _tokenize(self, text: str):
        """A simple tokenizer. A more advanced version could use regex or a library."""
        return text.lower().split()

    def learn_one(self, x: dict):
        """
        Updates the internal state based on a single document.
        This includes document count, document frequencies, and the vocabulary.
        """
        text = x.get(self.on, "")
        tokens = self._tokenize(text)

        # Increment total document count
        self.doc_count += 1
        
        # Update document frequencies for unique tokens in this document
        for token in set(tokens):
            self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
            # Add new tokens to the vocabulary
            if token not in self.vocabulary:
                self.vocabulary[token] = self._next_vocab_idx
                self._next_vocab_idx += 1
        
        return self

    def transform_one(self, x: dict):
        """
        Transforms a single document into a TF-IDF sparse vector.
        """
        text = x.get(self.on, "")
        tokens = self._tokenize(text)
        
        if not tokens:
            return {}

        # 1. Calculate Term Frequencies (TF) for the current document
        token_counts = Counter(tokens)
        total_tokens_in_doc = len(tokens)
        
        tfidf_vector = {}
        
        for token, count in token_counts.items():
            # Only generate features for words we have learned in the vocabulary
            if token in self.vocabulary:
                # Term Frequency calculation
                tf = count / total_tokens_in_doc
                
                # 2. Calculate Inverse Document Frequency (IDF) using the global state
                # We use a standard smoothed IDF formula to prevent division by zero
                # and to moderate the weight of rare words.
                # Formula: log((N+1) / (df+1)) + 1
                # N = total documents seen, df = documents containing the term
                df = self.doc_freqs.get(token, 0)
                idf = math.log((self.doc_count + 1) / (df + 1)) + 1
                
                # 3. Combine to get TF-IDF score
                feature_index = self.vocabulary[token]
                tfidf_vector[feature_index] = tf * idf
        
        return tfidf_vector
        
    def clone(self, include_attributes=True):
        return TFIDF_Calc(on=self.on)
        
    def _get_text(self, x):
        return x.get(self.on, "") if self.on is not None else x
        
    def _tokenize(self, text):
        if isinstance(text, str):
            return text.lower().split()
        return [str(text).lower()]

class TransformUnion_calc:
    """
    A custom implementation of an online transformer union.
    
    This class takes multiple transformer objects and applies them in parallel.
    It combines their resulting feature dictionaries into a single, larger one,
    ensuring feature keys are unique by prefixing them with the transformer's name.
    """
    def __init__(self, *transformers):
        """
        Args:
            *transformers: A variable number of transformer objects. Each object
                         is expected to have `learn_one`, `transform_one` methods,
                         and an `on` attribute to identify its input field.
        """
        self.transformers = transformers

    def learn_one(self, x: dict):
        """Calls `learn_one` on each internal transformer."""
        for transformer in self.transformers:
            transformer.learn_one(x)
        return self

    def transform_one(self, x: dict):
        """
        Calls `transform_one` on each transformer and merges the results.
        
        Feature keys are made unique by prefixing them with the name of the
        field the transformer operated on (e.g., 'summary_0', 'description_15').
        """
        combined_features = {}
        for transformer in self.transformers:
            # Get the sparse feature vector from the individual transformer
            individual_features = transformer.transform_one(x)
            
            # Get the name of the feature field to use as a prefix
            prefix = transformer.on
            
            # Merge the features, creating a new, unique key for each
            for key, value in individual_features.items():
                new_key = f"{prefix}_{key}"
                combined_features[new_key] = value
                
        return combined_features


# This class inherits from the MyOnlineModel we just wrote.
# This means it gets all its methods (_transform_features, process_one_issue, etc.) automatically.
class MyNaiveBayesModel(MyOnlineModel):
    """
    A simple online classifier using the Multinomial Naive Bayes algorithm.
    It relies entirely on the parent MyOnlineModel for its functionality.
    """
    def __init__(self, alpha=0.1):
        # --- Step A: Initialize the parent class ---
        # `super()` refers to the parent class (MyOnlineModel).
        # We call its __init__ method to set up the feature pipeline,
        # the accuracy trackers, and everything else.
        super().__init__()
        
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
    def __init__(self, alpha=0.1, delta=0.15):
        # --- Step A: Initialize the parent class, but this time passing the delta ---
        # Because we provide a 'drift_delta', the parent's __init__ method
        # will now create an instance of the ADWIN detector.
        super().__init__(drift_delta=delta)
        
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
            classifier=linear_model.PAClassifier(C=0.01, mode=1)
        )
        # Note: We don't need to override process_one_issue. The parent's default works perfectly.


### ========================================================== ###
###           MODEL : Softmax Regression CLASSIFIER           ###
### ========================================================== ###
class SoftmaxModel(MyOnlineModel):
    def __init__(self, drift_delta=0.05, **kwargs):
        # We DO pass drift_delta to the parent because this model needs it.
        super().__init__(drift_delta=drift_delta, **kwargs)
        
        self.name = "Softmax"
        # PassiveAggressiveClassifier is a linear model, wrapped in OneVsRest
        # to handle the multi-class assignment problem.
        self.ml_model = multiclass.OneVsRestClassifier(
            classifier=linear_model.SoftmaxRegression(
                optimizer=optim.Adam(0.01),
                # loss=lo.Hinge()
            )
        )
        # Note: We don't need to override process_one_issue. The parent's default works perfectly.


# This class inherits all the base functionality from MyOnlineModel
#Adaboost 10 --> 5 
class MyEnhancedModel(MyOnlineModel):
    """
    An advanced online classifier using an AdaBoost ensemble.
    
    It features two major enhancements:
    1. A custom prediction method that weights recently active developers more heavily.
    2. A smart drift-handling mechanism that only resets the worst-performing base models.
    """
    def __init__(self, alpha=0.05, delta=0.05, n_models=10):
        # --- Step A: Initialize the parent class ---
        # We pass the drift_delta to the parent to ensure the ADWIN detector is created.
        super().__init__(drift_delta=delta)
        
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
            self.drift_detector.update(self.running_metrics['accuracy'].get())
            
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









# ==============================================================================
# Part 3: Experiment Execution Logic (from s3_run_models.py)
# This section contains the function that runs an experiment for one model
# on one project.
# ==============================================================================
def run_single_experiment(model_class, project_name):
    """
    Executes a full online learning simulation for one model on one project.
    
    MODIFIED: This function now returns the final accuracy value.
    """
    
    # Load name mapping if it exists
    name_mapping = load_name_mapping(project_name)
    if name_mapping and hasattr(model, 'name_mapping'):
        model.name_mapping = name_mapping
    # --- Part A: Announce the start of the experiment ---
    temp_model_for_name = model_class()
    model_name = temp_model_for_name.name
    # (Print statements are commented out for cleaner parallel execution logs)
    print(f"--- Starting Experiment: {model_name} on {project_name} ---")

    # --- Part B: Load the project's data ---
    data_file_path = os.path.join(DATA_FOLDER, f"{project_name}.csv")
    try:
        raw_data_df = pd.read_csv(data_file_path)
    except FileNotFoundError:
        print(f"!!! ERROR: Data file not found for {project_name}. Skipping. !!!")
        return None # Return None to indicate failure

    # --- Step C: Preprocess the data ---
    features_stream, labels_stream = my_preprocess_data(raw_data_df)

    # --- Step D: Initialize a fresh model instance ---
    model = model_class()

    # --- Step E: Run the main online learning loop ---
    for i, (issue_features, true_label) in enumerate(zip(features_stream, labels_stream)):
        model.process_one_issue(
            issue_index=i,
            issue_features=issue_features,
            issue_label=true_label
        )

    # --- Step F: Report the final performance summary ---
    final_accuracy = model.running_metrics['accuracy'].get()
    total_drifts = len(model.detected_drifts_at)
    
    # This print statement can be helpful but will clutter the logs in parallel runs.
    # You can uncomment it for debugging.
    # print(f"--- Summary for {model_name} on {project_name}: Accuracy={final_accuracy:.4f}, Drifts={total_drifts} ---")

    # --- Step G: Save the detailed results to a JSON file ---
    output_directory = os.path.join(RESULTS_FOLDER, model.name)
    os.makedirs(output_directory, exist_ok=True)
    results_file_path = os.path.join(output_directory, f"{project_name}.json")
    results_data = model.get_results()
    with open(results_file_path, 'w') as f:
        json.dump(results_data, f, indent=4)
    
    
    save_model(model, project_name)
    # --- NEW AND IMPORTANT PART: Return the final accuracy ---
    return final_accuracy


def print_summary_table(summary_data, model_classes):
    """
    Prints a formatted summary table of final model accuracies.

    Args:
        summary_data (dict): A dictionary where keys are project names and values
                             are dicts of {model_name: accuracy}.
        model_classes (list): The list of model classes used to define the column order.
    """
    print("\n" + "="*120)
    print(" " * 45 + "FINAL ACCURACY SUMMARY")
    print("="*120)

    # Get model names in a fixed order for the table header
    model_names = [m().name for m in model_classes]
    
    # --- Print Header ---
    project_col_width = 15
    model_col_width = 28 # Wide enough for "My_Super_Enhanced_Model"

    header = f"{'Project':<{project_col_width}}"
    for name in model_names:
        header += f"{name:<{model_col_width}}"
    print(header)
    print("-" * len(header))

    # --- Print Data Rows ---
    for project_name, model_accuracies in sorted(summary_data.items()):
        row_str = f"{project_name:<{project_col_width}}"
        for model_name in model_names:
            # Get the accuracy for the current model, or 'N/A' if it's missing or None
            accuracy = model_accuracies.get(model_name)
            
            if isinstance(accuracy, float):
                formatted_acc = f"{accuracy * 100:.2f}%"
            else:
                formatted_acc = "N/A (Failed)" # Handle runs that returned None
            
            row_str += f"{formatted_acc:<{model_col_width}}"
        print(row_str)
        
    print("="*120)


def save_model(model, project_name):
    """Saves a trained model to disk for later use."""
    os.makedirs(MODEL_SAVE_FOLDER, exist_ok=True)
    save_path = os.path.join(MODEL_SAVE_FOLDER, f"{model.name}_{project_name}.pkl")
    
    # Save both the model and its name_mapping
    save_data = {
        'model': model,
        'name_mapping': model.name_mapping if hasattr(model, 'name_mapping') else []
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"Model saved to {save_path}")

def load_model(model_class, project_name):
    """Loads a previously trained model from disk."""
    model_name = model_class().name
    load_path = os.path.join(MODEL_SAVE_FOLDER, f"{model_name}_{project_name}.pkl")
    try:
        with open(load_path, 'rb') as f:
            save_data = pickle.load(f)
            model = save_data['model']
            # Restore the name_mapping if it exists
            if 'name_mapping' in save_data:
                model.name_mapping = save_data['name_mapping']
            print(f"Model loaded from {load_path}")
            return model
    except FileNotFoundError:
        print(f"No saved model found at {load_path}")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
def run_deployment_simulation(training_project: str, test_file_path: str, use_pretrained=True):
    """
    Simulates a real-world deployment scenario with interactive feedback.
    
    Args:
        training_project: The project name to use for training (if needed)
        test_file_path: Path to CSV file with new issues to process
        use_pretrained: If True, tries to load a pre-trained model first
    """
    print("="*80)
    print(" " * 20 + "DEPLOYMENT SIMULATION STARTED")
    print("="*80)

    model = None
    name_mapping = load_name_mapping(training_project)

    
    # --- Step 1: Try to load pre-trained model ---
    if use_pretrained:
        model = load_model(MySuperEnhancedModel, training_project)
        if model and name_mapping:
            model.name_mapping = name_mapping
    
    # --- Step 2: If no pre-trained model, train a new one ---
    if model is None:
        print(f"\n[PHASE 1] Pre-training model on historical data from project: '{training_project}'...")
        
        # Initialize a fresh model instance
        model = MySuperEnhancedModel()
        
        # Load and preprocess the training data
        train_file_path = os.path.join(DATA_FOLDER, f"{training_project}.csv")
        try:
            train_df = pd.read_csv(train_file_path)
        except FileNotFoundError:
            print(f"!!! ERROR: Training data file not found at {train_file_path}. Halting. !!!")
            return
            
        features_stream, labels_stream = my_preprocess_data(train_df)
        
        # Run the standard simulation loop to train the model
        for i, (features, label) in enumerate(zip(features_stream, labels_stream)):
            model.process_one_issue(issue_index=i, issue_features=features, issue_label=label)
            
        print(f"Model pre-training complete. Final accuracy on training data: {model.running_metrics['accuracy'].get():.4f}")
        
        # Save the trained model for future use
        save_model(model, training_project)
    else:
        print("\nUsing pre-trained model. Skipping training phase.")
    
    print("\nThe model is now 'live' and ready for new issues.")
    print(f"Current model accuracy: {model.running_metrics['accuracy'].get():.2%}\n")

    # --- Step 3: Process new issues interactively ---
    print(f"\n[PHASE 2] Processing new issues from test file: '{test_file_path}'...")
    
    try:
        test_df = pd.read_csv(test_file_path)
    except FileNotFoundError:
        print(f"!!! ERROR: Test data file not found at {test_file_path}. Halting. !!!")
        return

    test_features_stream, _ = my_preprocess_data(test_df) # We don't need the labels here

    for i, issue_features in enumerate(test_features_stream):
        print("\n" + "-"*60)
        print(f"--- A new issue has arrived (Issue #{i + 1}) ---")
        print(f"  Summary: {issue_features.get('summary', 'N/A')}")
        
        # 1. PREDICT: Get recommendations from the model
        recommendations, feature_vector = model.predict_recommendations(issue_features, top_n=3)
        
        print("\n  Model Recommendations:")
        if recommendations:
            for rank, dev in enumerate(recommendations):
                print(f"    {rank + 1}. {dev}")
        else:
            print("    The model has no recommendations at this time.")

        # 2. GET FEEDBACK: Ask the user for the true assignment
        true_label = None
        while not true_label:
            true_label = input("\n  ACTION: Please enter the name of the developer this issue was assigned to: ").strip()
            if not true_label:
                print("  Input cannot be empty. Please try again.")

        # 3. LEARN: Update the model with the confirmed assignment
        # We use the total number of training issues + current index for the 'issue_index'
        learning_index = len(model.last_assignees) + i
        model.learn_from_assignment(feature_vector, true_label, issue_index=learning_index)
        
        # Print updated accuracy
        current_accuracy = model.running_metrics['accuracy'].get()
        print(f"\n  Feedback received. Model has learned from assigning the issue to '{true_label}'.")
        print(f"  Updated model accuracy: {current_accuracy:.2%}")
        save_model(model,training_project)

    print("\n" + "="*80)
    print(" " * 22 + "DEPLOYMENT SIMULATION FINISHED")
    print("="*80)

def load_name_mapping(project_name):
    """Loads the name mapping for a project"""
    mapping_file = os.path.join(DATA_FOLDER, f"{project_name}_name_mapping.csv")
    try:
        return pd.read_csv(mapping_file, header=None)[0].tolist()
    except FileNotFoundError:
        return []
# ==============================================================================
# Part 4: Main Execution Block
# ==============================================================================
if __name__ == '__main__':
    # We use argparse to allow the GUI to pass parameters to this script
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Jira Issue Assigner Models.")
    parser.add_argument(
        '--mode', 
        type=str, 
        required=True, 
        choices=['experiment', 'deploy'],
        help="The mode to run the script in: 'experiment' or 'deploy'."
    )
    parser.add_argument(
        '--train_project', 
        type=str, 
        default="AMBARI", 
        help="For deploy mode: the project to use for training the model."
    )
    parser.add_argument(
        '--test_file', 
        type=str, 
        default="My Scrum Project_test.csv", 
        help="For deploy mode: the CSV file containing new issues to process."
    )
    parser.add_argument(
        '--no_pretrained', 
        action='store_true',
        help="For deploy mode: add this flag to force retraining from scratch."
    )
    
    args = parser.parse_args()

    # --- CHOOSE YOUR MODE ---
    MODE = args.mode

    if MODE == 'experiment':
        print("Script execution started in EXPERIMENT mode.")
        models_to_test = [MyNaiveBayesModel]
        # Dynamically find all project CSVs in the data folder
        try:
            projects_to_process = [f[:-4] for f in os.listdir(DATA_FOLDER) if f.endswith(".csv") and "name_mapping" not in f]
        except FileNotFoundError:
            print(f"ERROR: Data folder '{DATA_FOLDER}' not found. Cannot run experiments.")
            projects_to_process = []
            
        if projects_to_process:
            experiment_tasks = [(m, p) for m in models_to_test for p in projects_to_process]
            num_workers = min(mp.cpu_count(), 6)
            print(f"Starting experiments in PARALLEL using {num_workers} worker processes...")
            with mp.Pool(processes=num_workers) as pool:
                results_list = pool.starmap(run_single_experiment, experiment_tasks)
            print("\nAll experiments have finished.")
            
            # Collate and print results
            summary_data = defaultdict(dict)
            for task, final_accuracy in zip(experiment_tasks, results_list):
                model_class, project_name = task
                model_name = model_class().name
                summary_data[project_name][model_name] = final_accuracy
            print_summary_table(summary_data, models_to_test)
        else:
            print("No projects found to process. Halting execution.")

    elif MODE == 'deploy':
        print("Script execution started in DEPLOYMENT SIMULATION mode.")
        
        # Use parameters passed from the GUI
        PROJECT_FOR_TRAINING = args.train_project
        DEPLOYMENT_TEST_FILE = args.test_file
        
        # The DATA_FOLDER for deployment mode is where the test CSV is located
        test_file_path = os.path.join(DATA_FOLDER, DEPLOYMENT_TEST_FILE)
        
        # The 'use_pretrained' flag is True unless --no_pretrained is passed
        use_pretrained_model = not args.no_pretrained

        run_deployment_simulation(
            training_project=PROJECT_FOR_TRAINING,
            test_file_path=test_file_path,
            use_pretrained=use_pretrained_model
        )
    
    # This keeps the terminal window open after the script finishes on Windows
    # so you can read the output.
    if os.name == 'nt':
        os.system('pause')