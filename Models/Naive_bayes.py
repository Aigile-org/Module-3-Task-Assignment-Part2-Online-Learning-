
from running_models import OnlineInterface
import pandas as pd
from collections import defaultdict
from itertools import repeat
import math
from collections import defaultdict
from typing import Dict, Any, Optional, Set
import numpy as np
class NaiveBayesModel(OnlineInterface):
    def __init__(self, alpha=1.0):
        super().__init__()
        
        # Must use alpha=1.0 to match River's default
        self.alpha = alpha  
        self.name = "Naive_Bayes"
        
        # EXACTLY matching River's internal data structures
        self.class_counts = defaultdict(int)
        self.feature_counts = defaultdict(lambda: defaultdict(float))
        self.class_totals = defaultdict(float)
        self.total_count = 0
        self._class_priors = defaultdict(float)
        self._feature_log_probs = defaultdict(lambda: defaultdict(float))
        
        # Critical River-specific attributes
        self._n_features = 0  # Tracks feature count exactly like River
        self._seen_features = set()
        self._min_float = np.finfo(float).tiny  # Smallest positive float
        
        # River's prediction caching
        self._last_features = None
        self._last_prediction = None
    
    def _update_model(self, feature_vector: Dict[str, float], true_label: str) -> None:
        """EXACT duplicate of River's update logic"""
        # River first converts to dense representation
        features = defaultdict(float)
        features.update(feature_vector)
        
        # Update class counts
        self.class_counts[true_label] += 1
        self.total_count += 1
        
        # Track NEW features (River does this before updating counts)
        new_features = set(features.keys()) - self._seen_features
        if new_features:
            self._n_features += len(new_features)
            self._seen_features.update(new_features)
            # Invalidate cached probabilities
            self._feature_log_probs.clear()
        
        # Update feature counts using ABSOLUTE values (River does this)
        feature_sum = 0.0
        for feature, value in features.items():
            abs_value = abs(value)  # Critical - River uses absolute values
            self.feature_counts[true_label][feature] += abs_value
            feature_sum += abs_value
        self.class_totals[true_label] += feature_sum
        
        # Force recalculation of probabilities for this class
        if true_label in self._feature_log_probs:
            del self._feature_log_probs[true_label]
    
    def _calculate_log_probs(self, class_label: str) -> None:
        """EXACT probability calculation matching River"""
        # Calculate class prior
        self._class_priors[class_label] = self.class_counts[class_label] / self.total_count
        
        # Calculate denominator (River does this per class)
        denominator = self.class_totals[class_label] + self.alpha * self._n_features
        
        # Calculate log probabilities with River's smoothing
        self._feature_log_probs[class_label] = defaultdict(lambda: math.log(self.alpha / denominator))
        for feature, count in self.feature_counts[class_label].items():
            prob = (count + self.alpha) / denominator
            self._feature_log_probs[class_label][feature] = math.log(max(prob, self._min_float))
    
    def _predict_one(self, feature_vector: Dict[str, float]) -> Optional[str]:
        """EXACT prediction logic from River"""
        if self.total_count == 0:
            return None
            
        # Check prediction cache (River does this)
        if feature_vector == self._last_features:
            return self._last_prediction
            
        # Convert to dense features (like River)
        features = defaultdict(float)
        features.update(feature_vector)
        
        best_class = None
        best_log_prob = -math.inf
        
        for class_label in self.class_counts:
            # Lazy calculation of probabilities (like River)
            if class_label not in self._feature_log_probs:
                self._calculate_log_probs(class_label)
            
            # Start with log prior
            try:
                log_prob = math.log(self._class_priors[class_label])
            except ValueError:
                log_prob = math.log(self._min_float)
            
            # Add feature log probs (using ABSOLUTE values)
            for feature, value in features.items():
                log_prob += abs(value) * self._feature_log_probs[class_label][feature]
            
            # River's exact comparison logic
            if log_prob > best_log_prob or (log_prob == best_log_prob and (best_class is None or class_label < best_class)):
                best_log_prob = log_prob
                best_class = class_label
        
        # Cache prediction (like River)
        self._last_features = feature_vector
        self._last_prediction = best_class
        
        return best_class
    
    def _predict_and_learn(self, feature_vector: Dict[str, float], true_label: str) -> Optional[str]:
        """EXACT predict-then-learn cycle from River"""
        # Predict first
        prediction = self._predict_one(feature_vector)
        
        # Then learn (this order matters!)
        self._update_model(feature_vector, true_label)
        
        return prediction

