from collections import defaultdict
import random
import math
from typing import Dict, Any
import numpy as np


class MyAdaBoostClassifier:
    def __init__(self, model, n_models: int = 10, seed: int = 42):
        self.model = model
        self.n_models = n_models
        self.seed = seed
        random.seed(seed)
        
        # Initialize models and weights (EXACTLY like River does)
        self.models = [self.model.clone() for _ in range(n_models)]
        self.weights = [1.0] * n_models
        
        # These attributes must exist for your drift detection
        self.correct_weight = [0.0] * n_models
        self.wrong_weight = [0.0] * n_models
        
        # Tracking for online boosting
        self._beta = [1.0] * n_models
        self._sample_weights = [1.0] * n_models
        self._current_model = 0
        self._n_training_samples = 0
        
        # Multi-class support
        self.classes_ = set()

    def learn_one(self, x: Dict, y: Any, sample_weight: float = 1.0) -> None:
        """Update the model with a single sample (matches River exactly)."""
        # Track seen classes
        self.classes_.add(y)
        
        # Weighted learning (handles models that don't support sample_weight)
        current_weight = self._sample_weights[self._current_model] * sample_weight
        
        # Special handling for models that don't accept sample_weight
        if hasattr(self.models[self._current_model], 'learn_one'):
            try:
                self.models[self._current_model].learn_one(x, y, sample_weight=current_weight)
            except TypeError:
                # If model doesn't support sample_weight, learn multiple times
                effective_count = max(1, int(round(current_weight)))
                for _ in range(effective_count):
                    self.models[self._current_model].learn_one(x, y)
        
        # Get prediction and update weights
        y_pred = self.models[self._current_model].predict_one(x)
        incorrect = y_pred != y
        
        # Update correct/wrong weights (for drift detection)
        if incorrect:
            self.wrong_weight[self._current_model] += current_weight
        else:
            self.correct_weight[self._current_model] += current_weight
        
        # Update sample weights for next models
        if incorrect:
            error = max(self.wrong_weight[self._current_model] / 
                      (self.correct_weight[self._current_model] + self.wrong_weight[self._current_model]), 1e-10)
            beta = error / (1 - error)
            self._beta[self._current_model] = beta
            
            for j in range(self._current_model + 1, self.n_models):
                self._sample_weights[j] *= beta
        
        # Move to next model
        self._current_model = (self._current_model + 1) % self.n_models
        self._n_training_samples += 1
        
        # Update model weights periodically
        if self._n_training_samples % 100 == 0:
            self._update_weights()

    def _update_weights(self) -> None:
        """Update model weights based on performance."""
        for i in range(self.n_models):
            total = self.correct_weight[i] + self.wrong_weight[i]
            if total > 0:
                error = self.wrong_weight[i] / total
                self.weights[i] = math.log((1 - error) / max(error, 1e-10))
            else:
                self.weights[i] = 1.0

    def predict_one(self, x: Dict) -> Any:
        """Predict the label for a single sample (matches River exactly)."""
        if not self.classes_:
            return None
            
        # Binary classification
        if len(self.classes_) == 2:
            weighted_sum = 0.0
            for model, weight in zip(self.models, self.weights):
                pred = model.predict_one(x)
                if pred == 1:  # Positive class
                    weighted_sum += weight
                else:  # Negative class
                    weighted_sum -= weight
            return 1 if weighted_sum >= 0 else 0
        else:  # Multi-class
            return self.predict_proba_one(x).most_common(1)[0][0]

    def predict_proba_one(self, x: Dict) -> Dict[Any, float]:
        """Predict class probabilities (matches River exactly)."""
        probas = defaultdict(float)
        total_weight = sum(self.weights)
        
        if not self.classes_ or total_weight <= 0:
            return probas
            
        # Get weighted votes from all models
        for model, weight in zip(self.models, self.weights):
            if hasattr(model, 'predict_proba_one'):
                model_probas = model.predict_proba_one(x)
                for cls, prob in model_probas.items():
                    probas[cls] += weight * prob
            else:
                pred = model.predict_one(x)
                probas[pred] += weight
                
        # Normalize
        for cls in probas:
            probas[cls] /= total_weight
            
        return probas

    def clone(self):
        """Create a fresh copy (matches River's clone behavior)."""
        return MyAdaBoostClassifier(
            model=self.model.clone(),
            n_models=self.n_models,
            seed=self.seed
        )
