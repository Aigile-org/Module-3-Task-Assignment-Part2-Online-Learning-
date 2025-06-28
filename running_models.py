# ==============================================================================
# Part 0: Imports and Configuration
# All necessary libraries are imported at the top.
# ==============================================================================
import os
from collections import defaultdict
import multiprocessing as mp
import random
from Utils.utilities import run_deployment_simulation, run_single_experiment, print_summary_table
from river import naive_bayes, multiclass, ensemble, drift, metrics,  linear_model ,tree, optim
from Models.OnlineInterface import OnlineInterface
TEST = True
if not TEST:
    DATA_FOLDER = "data/"
else:
    DATA_FOLDER = "deploy and test\csvs"    
RESULTS_FOLDER = "results_new_1/"
MODEL_SAVE_FOLDER = "saved_models/"


# This class inherits from the OnlineInterface we just wrote.
# This means it gets all its methods (_transform_features, process_one_issue, etc.) automatically.
class NaiveBayesModel(OnlineInterface):
    """
    A simple online classifier using the Multinomial Naive Bayes algorithm.
    It relies entirely on the parent OnlineInterface for its functionality.
    """
    def __init__(self, alpha=0.1):
        # --- Step A: Initialize the parent class ---
        # `super()` refers to the parent class (OnlineInterface).
        # We call its __init__ method to set up the feature pipeline,
        # the accuracy trackers, and everything else.
        super().__init__()
        
        # --- Step B: Define the specific ML model for this class ---
        # We now assign a specific algorithm to the `self.ml_model` placeholder
        # that was left empty in the parent class.
        self.ml_model = naive_bayes.MultinomialNB(alpha=alpha)
        
        # --- Step C: Give our model a name for reports and file paths ---
        self.name = "My_Naive_Bayes"

class NaiveBayesWithDrift(OnlineInterface):
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
        # Calling `super().process_one_issue(...)` runs the original method from OnlineInterface.
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

class HoeffdingAdaptiveTreeModel(OnlineInterface):
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

class LeveragingBaggingModel(OnlineInterface):
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

class PassiveAggressiveModel(OnlineInterface):
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

class SoftmaxModel(OnlineInterface):
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

class AdaboostWithDrift(OnlineInterface):
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
        self.name = "Adaboost"

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
     

class EnhancedAdaboost(OnlineInterface):
    """
    A "super" enhanced version of the original AdaBoost model.
    It now includes methods for a real-world deployment scenario where
    prediction and learning are separate steps.
    """
    def __init__(self, alpha=0.1, delta=0.15, n_models=10, name_mapping=None,**kwargs):
        # This super().__init__() call correctly initializes the pipeline
        # from OnlineInterface, which looks for 'summary', 'description', etc.
        super().__init__(drift_delta=delta, **kwargs)
        self.name_mapping = name_mapping or []

        
        self.name = "Enhanced_Adaboost"
        
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
        models_to_test = [NaiveBayesModel]
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