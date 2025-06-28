import json
import logging
import os
import boto3
import pickle
import sys
from typing import Dict, Any, Optional
from collections import defaultdict
import random
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS clients
s3_client = boto3.client('s3')

# Try to import River with graceful fallback
RIVER_AVAILABLE = False
try:
    from river import feature_extraction, compose, naive_bayes, multiclass, ensemble, drift, metrics, linear_model, tree
    RIVER_AVAILABLE = True
    logger.info("✅ River imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ River import failed: {e}")
    RIVER_AVAILABLE = False

# ==============================================================================
# Model Class Definitions (needed for pickle unpickling)
# ==============================================================================

class OnlineInterface:
    """
    A blueprint for an online machine learning model for issue assignment.
    
    This class provides the core framework for:
    1. Feature extraction using TF-IDF.
    2. Tracking progressive accuracy.
    3. Detecting concept drift with ADWIN.
    """
    def __init__(self, drift_delta=None):
        
        # --- Part A: Define which features to use ---
        feature_names = ["summary", "description", "labels", "components_name", "priority_name", "issue_type_name"]

        # --- Part B: Set up the Drift Detector ---
        # The drift detector is optional. We only create it if a 'delta' value is given.
        if drift_delta is not None and RIVER_AVAILABLE:
            self.drift_detector = drift.ADWIN(delta=drift_delta)
        else:
            self.drift_detector = None
        
        # --- Part C: Create the Feature Engineering Pipeline ---
        # This is the core of our feature preparation.
        # We create a list where each item is a TF-IDF transformer for one feature.
        # The `on=name` tells each transformer which key to look for in the input dictionary.
        list_of_transformers = []
        if RIVER_AVAILABLE:
            for name in feature_names:
                list_of_transformers.append(feature_extraction.TFIDF(on=name))

            # `TransformerUnion` combines the outputs of all individual transformers
            # into a single, large feature vector. The '*' unpacks our list into arguments.
            self.feature_pipeline = compose.TransformerUnion(*list_of_transformers)
        else:
            self.feature_pipeline = None

        # --- Part D: Initialize containers for results ---
        self.accuracies_over_time = []  # To store accuracy at each step
        self.detected_drifts_at = []    # To store the index 'i' where drift occurs
        if RIVER_AVAILABLE:
            self.running_accuracy = metrics.Accuracy() # The river object to calculate rolling accuracy
        else:
            self.running_accuracy = None
        
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
        if not self.feature_pipeline:
            return single_issue_dict
            
        # First, let the pipeline learn from the raw data.
        self.feature_pipeline.learn_one(single_issue_dict)
        
        # Then, get the transformed numerical vector.        # First, let the pipeline learn from the raw data.
        self.feature_pipeline.learn_one(single_issue_dict)
        
        # Then, get the transformed numerical vector.
        transformed_vector = self.feature_pipeline.transform_one(single_issue_dict)
        
        return transformed_vector
        
    def _update_accuracy(self, true_label, predicted_label):
        """Updates the rolling accuracy and stores its current value."""
        if not self.running_accuracy:
            return
            
        # Update the river metric object with the latest result        self.running_accuracy.update(true_label, predicted_label)
        
        # Append the new overall accuracy to our list for later plotting
        self.accuracies_over_time.append(self.running_accuracy.get())


class EnhancedAdaboost(OnlineInterface):
    """
    A "super" enhanced version of the original AdaBoost model.
    It now includes methods for a real-world deployment scenario where
    prediction and learning are separate steps.
    """
    def __init__(self, alpha=0.1, delta=0.15, n_models=10, name_mapping=None, **kwargs):
        # This super().__init__() call correctly initializes the pipeline
        # from OnlineInterface, which looks for 'summary', 'description', etc.
        super().__init__(drift_delta=delta, **kwargs)
        self.name_mapping = name_mapping or []

        self.name = "Enhanced_Adaboost"
        
        if RIVER_AVAILABLE:
            base_learner = multiclass.OneVsRestClassifier(naive_bayes.MultinomialNB(alpha=alpha))
            self.ml_model = ensemble.AdaBoostClassifier(model=base_learner, n_models=n_models, seed=42)
        else:
            self.ml_model = None
        
        self.n_models = n_models
        if RIVER_AVAILABLE:
            self.base_model_accuracies = [metrics.Accuracy() for _ in range(self.n_models)]
        else:
            self.base_model_accuracies = []
        self.recency_window = 100
        self.last_assignees = []

    def predict_recommendations(self, issue_features: dict, top_n=3):
        """
        Step 1 of Deployment: Predicts assignees for a new issue with an unknown label.
        
        Returns:
            - A list of top_n recommended assignees (with actual names if mapping exists)
            - The generated feature_vector (to be used later in the learning step)
        """
        if not self.ml_model:
            return [], issue_features
            
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
        if not self.ml_model:
            return None
            
        # --- Part 1: Update the core model ---
        self.ml_model.learn_one(feature_vector, true_label)
        
        # --- Part 2: Update recency heuristic list ---
        self.last_assignees.append(true_label)
        if len(self.last_assignees) > self.recency_window:
            self.last_assignees.pop(0)

        # --- Part 3: Update accuracy metrics ---
        # Generate a prediction to update accuracy metrics
        prediction = self._predict_for_accuracy(feature_vector)
        if prediction is not None and self.running_accuracy:
            self._update_accuracy(true_label, prediction)
        
        # Update base model accuracies
        for i, model in enumerate(self.ml_model.models if hasattr(self.ml_model, 'models') else []):
            if i < len(self.base_model_accuracies):
                if (base_pred := model.predict_one(feature_vector)) is not None:
                    self.base_model_accuracies[i].update(true_label, base_pred)

        # Update and check the ADWIN drift detector
        if self.drift_detector is not None:
            is_correct = 1 if prediction == true_label else 0
            self.drift_detector.update(is_correct)
            
            if self.drift_detector.drift_detected:
                self.detected_drifts_at.append(issue_index)
                logger.info(f"Drift detected! Applying probabilistic reset...")
                if hasattr(self.ml_model, 'models'):
                    for i in range(self.n_models):
                        if i < len(self.base_model_accuracies):
                            accuracy = self.base_model_accuracies[i].get()
                            if random.random() < (1.0 - accuracy):
                                self.ml_model.models[i] = self.ml_model.model.clone()
                                self.ml_model.correct_weight[i], self.ml_model.wrong_weight[i] = 0.0, 0.0
                                self.base_model_accuracies[i] = metrics.Accuracy() if RIVER_AVAILABLE else None
        
        return prediction

    def _predict_for_accuracy(self, feature_vector):
        """Helper method to generate predictions for accuracy tracking"""
        if not self.ml_model:
            return None
        probabilities = self.ml_model.predict_proba_one(feature_vector)
        if probabilities:
            return max(probabilities, key=probabilities.get)
        return None

    # Lambda-compatible methods
    def predict_assignment(self, issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Lambda-compatible prediction method.
        Returns top 3 recommended assignees.
        """
        try:
            recommendations, feature_vector = self.predict_recommendations(issue_data, top_n=3)
            
            return {
                "recommendations": recommendations if recommendations else ["unassigned"],
                "model_type": self.name
            }
        except Exception as e:
            logger.error(f"Prediction error in EnhancedAdaboost: {str(e)}")
            return {
                "recommendations": ["unassigned"],
                "error": str(e),
                "model_type": self.name
            }

    def partial_fit(self, issue_data: Dict[str, Any], assignee: str) -> Dict[str, Any]:
        """
        Lambda-compatible online learning method.
        Returns a dictionary with learning details.
        """
        try:
            # Transform features first
            feature_vector = self._transform_features(issue_data)
            
            # Learn from the assignment (use a dummy issue index)
            issue_index = len(self.last_assignees)
            result = self.learn_from_assignment(feature_vector, assignee, issue_index)
            
            return {
                "success": True,
                "message": f"Model learned from assignment: {assignee}",
                "model_accuracy": self.running_accuracy.get() if self.running_accuracy else 0.0,
                "total_assignments": len(self.last_assignees),
                "prediction_result": result
            }
        except Exception as e:
            logger.error(f"Learning error in EnhancedAdaboost: {str(e)}")
            return {
                "success": False,
                "error": str(e),                "assignee": assignee
            }

def download_from_s3(bucket, key, local_path):
    """
    Download a file from S3 to a local path.
    
    Args:
        bucket (str): S3 bucket name
        key (str): S3 object key
        local_path (str): Local file path to save the downloaded file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Downloading {key} from bucket {bucket} to {local_path}")
        s3_client.download_file(bucket, key, local_path)
        logger.info(f"Successfully downloaded {key} to {local_path}")
        return True
    except ClientError as e:
        logger.error(f"Error downloading from S3: {e}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"Unexpected error downloading from S3: {e}", exc_info=True)
        return False

def load_model_from_s3():
    """Load the River model from S3 bucket using local file download."""
    
    bucket_name = "aigile-bucket"
    model_key = "Enhanced_Adaboost_AMBARI.pkl"
    local_model_path = "/tmp/model.pkl"  # Lambda /tmp directory
    
    try:
        logger.info(f"Loading model from S3: {bucket_name}/{model_key}")
        
        # Always download fresh model (no caching)
        logger.info("Loading fresh model instance (caching disabled)")
        
        # Download model file to local path
        download_success = download_from_s3(bucket_name, model_key, local_model_path)
        
        if not download_success:
            raise Exception("Failed to download model file from S3")# Load the model from local file
        logger.info(f"Loading model from local file: {local_model_path}")
        
        # CRITICAL FIX: Register our classes in __main__ module so pickle can find them
        # This solves the "Can't get attribute 'EnhancedAdaboost' on <module '__main__'" error
        logger.info("Registering model classes in __main__ module for pickle compatibility...")
        sys.modules['__main__'].OnlineInterface = OnlineInterface
        sys.modules['__main__'].EnhancedAdaboost = EnhancedAdaboost
        
        with open(local_model_path, 'rb') as f:
            loaded_data = pickle.load(f)
        
        # Handle different model file formats
        logger.info(f"Loaded data type: {type(loaded_data)}")
        
        if isinstance(loaded_data, dict):
            logger.info("Model file contains a dictionary, extracting model...")
            logger.info(f"Dictionary keys: {list(loaded_data.keys())}")
            
            # Try common dictionary keys for the model
            if 'model' in loaded_data:
                model_instance = loaded_data['model']
                logger.info(f"Found model in 'model' key, type: {type(model_instance)}")
            elif 'trained_model' in loaded_data:
                model_instance = loaded_data['trained_model']
                logger.info(f"Found model in 'trained_model' key, type: {type(model_instance)}")
            elif len(loaded_data) == 1:
                # If there's only one item, use it
                key = list(loaded_data.keys())[0]
                model_instance = loaded_data[key]
                logger.info(f"Found model in '{key}' key, type: {type(model_instance)}")
            else:
                # Try to find a EnhancedAdaboost object in the dictionary
                for key, value in loaded_data.items():
                    if hasattr(value, 'predict_assignment') or isinstance(value, EnhancedAdaboost):
                        model_instance = value
                        logger.info(f"Found model object in '{key}' key, type: {type(model_instance)}")
                        break
                else:
                    # If no model found, raise an error
                    raise Exception(f"Could not find model object in dictionary. Available keys: {list(loaded_data.keys())}")
                    
        elif hasattr(loaded_data, 'predict_assignment') or isinstance(loaded_data, EnhancedAdaboost):
            # Direct model object
            model_instance = loaded_data
            logger.info(f"Direct model object loaded, type: {type(model_instance)}")
        else:
            raise Exception(f"Unexpected model file format. Type: {type(loaded_data)}, has predict_assignment: {hasattr(loaded_data, 'predict_assignment')}")
        
        # Verify the model has the required methods
        if not hasattr(model_instance, 'predict_assignment'):
            logger.warning(f"Model type {type(model_instance)} doesn't have predict_assignment method")
            # Try to add the method if it's a EnhancedAdaboost but missing the method
            if isinstance(model_instance, EnhancedAdaboost):
                logger.warning("Model is EnhancedAdaboost but missing predict_assignment method")
          # Clean up the temporary file
        try:
            os.remove(local_model_path)
            logger.info("Cleaned up temporary model file")
        except Exception as e:
            logger.warning(f"Could not clean up temporary file: {e}")
        
        logger.info(f"✅ Model loaded successfully from local file! Type: {type(model_instance).__name__}")
        return model_instance
        
    except Exception as e:
        logger.error(f"Failed to load model from S3 using local file approach: {str(e)}")
        # Clean up the temporary file in case of error
        try:
            if os.path.exists(local_model_path):
                os.remove(local_model_path)
        except Exception:
            pass
        raise

def save_model_to_s3(model):
    """Save the updated model back to S3 using local file approach."""
    global model_last_modified
    
    bucket_name = "aigile-bucket"
    model_key = "Enhanced_Adaboost_AMBARI.pkl"
    local_model_path = "/tmp/model_save.pkl"  # Lambda /tmp directory
    
    try:
        logger.info(f"Saving updated model to S3: {bucket_name}/{model_key}")
        
        # Serialize the model to local file first
        logger.info(f"Serializing model to local file: {local_model_path}")
        with open(local_model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Upload the local file to S3
        logger.info(f"Uploading local file to S3...")
        s3_client.upload_file(
            local_model_path,
            bucket_name,
            model_key,
            ExtraArgs={'ContentType': 'application/octet-stream'}
        )
        
        # Clean up the temporary file
        try:
            os.remove(local_model_path)
            logger.info("Cleaned up temporary save file")
        except Exception as e:
            logger.warning(f"Could not clean up temporary save file: {e}")
                
        logger.info("✅ Model saved successfully to S3 using local file approach")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save model to S3 using local file approach: {str(e)}")
        # Clean up the temporary file in case of error
        try:
            if os.path.exists(local_model_path):
                os.remove(local_model_path)
        except Exception:
            pass
        return False

def lambda_handler(event, context):
    """River Lambda container handler with S3 model loading"""
    
    headers = {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Requested-With'
    }
    
    try:
        # Handle OPTIONS preflight
        if event.get('httpMethod') == 'OPTIONS':
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps({'message': 'CORS preflight successful'})
            }
        
        # Parse body
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event.get('body', {})
        
        action = body.get('action', 'health')
        logger.info(f"Processing action: {action}")
        
        if action == 'health':
            # Test model loading for health check
            try:
                model = load_model_from_s3()
                model_loaded = True
                model_type = type(model).__name__
            except Exception as e:
                logger.warning(f"Model loading failed in health check: {e}")
                model_loaded = False
                model_type = "unknown"
            
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps({
                    'status': 'healthy',
                    'message': 'River Lambda Container with S3 model loading',
                    'river_available': RIVER_AVAILABLE,
                    'model_loaded': model_loaded,
                    'model_type': model_type,
                    'bucket': 'aigile-bucket',
                    'model_file': 'Enhanced_Adaboost_AMBARI.pkl'
                })
            }
        
        elif action == 'predict':
            issue_data = body.get('issue', {})
            
            if not issue_data:
                return {
                    'statusCode': 400,
                    'headers': headers,
                    'body': json.dumps({
                        'error': 'Missing issue data',
                        'required_fields': ['issue']
                    })
                }
            
            try:
                # Load model from S3
                model = load_model_from_s3()
                
                # Use the model's predict_assignment method
                if hasattr(model, 'predict_assignment'):
                    result = model.predict_assignment(issue_data)
                    logger.info(f"Model prediction successful: {result}")                    
                    return {
                        'statusCode': 200,
                        'headers': headers,
                        'body': json.dumps({
                            'success': True,
                            'prediction': result,
                            'model_used': 'river_s3_model',
                            'model_type': type(model).__name__
                        })
                    }
                else:
                    # Model doesn't have the expected method
                    logger.error("Model doesn't have predict_assignment method")
                    return {
                        'statusCode': 500,
                        'headers': headers,
                        'body': json.dumps({
                            'success': False,
                            'error': 'Model loaded but missing predict_assignment method'
                        })
                    }
                    
            except Exception as e:
                logger.error(f"Model prediction failed: {str(e)}")
                return {
                    'statusCode': 500,
                    'headers': headers,
                    'body': json.dumps({
                        'success': False,
                        'error': f'Model prediction failed: {str(e)}'
                    })
                }
        
        elif action == 'partial_fit':
            assignee = body.get('assignee')
            issue_data = body.get('issue', {})
            
            if not assignee:
                return {
                    'statusCode': 400,
                    'headers': headers,
                    'body': json.dumps({
                        'error': 'Missing assignee',
                        'required_fields': ['assignee', 'issue']
                    })
                }
            
            if not issue_data:
                return {
                    'statusCode': 400,
                    'headers': headers,
                    'body': json.dumps({
                        'error': 'Missing issue data',
                        'required_fields': ['assignee', 'issue']
                    })
                }
            
            try:
                # Load model from S3
                model = load_model_from_s3()
                  # Use the model's partial_fit method
                if hasattr(model, 'partial_fit'):
                    result = model.partial_fit(issue_data, assignee)
                    logger.info(f"Model partial_fit successful: {result}")
                    
                    # Return success immediately, save model asynchronously
                    response_body = {
                        'success': True,
                        'message': f'Model updated with assignment: {assignee}',
                        'learning_result': result,
                        'learning_method': 'river_partial_fit'
                    }
                    
                    # Try to save model, but don't let it block the response
                    try:
                        save_success = save_model_to_s3(model)
                        response_body['model_saved'] = save_success
                        logger.info(f"Model saved to S3 successfully: {save_success}")
                    except Exception as save_error:
                        logger.warning(f"Model save failed, but partial_fit was successful: {save_error}")
                        response_body['model_saved'] = False
                        response_body['save_warning'] = str(save_error)
                    
                    return {
                        'statusCode': 200,
                        'headers': headers,
                        'body': json.dumps(response_body)
                    }
                else:
                    logger.warning("Model doesn't have partial_fit method")
                    return {
                        'statusCode': 200,
                        'headers': headers,
                        'body': json.dumps({
                            'success': False,
                            'message': 'Model loaded but does not support online learning (partial_fit)',
                            'model_type': type(model).__name__
                        })
                    }
                    
            except Exception as e:
                logger.error(f"Model partial_fit failed: {str(e)}")
                
                return {
                    'statusCode': 500,
                    'headers': headers,
                    'body': json.dumps({
                        'success': False,
                        'error': f'Failed to update model: {str(e)}',
                        'assignee': assignee
                    })
                }
        
        else:
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps({
                    'error': f'Unsupported action: {action}',
                    'supported_actions': ['predict', 'partial_fit', 'health']
                })
            }
    
    except Exception as e:
        logger.error(f"Lambda handler error: {str(e)}")
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({
                'error': 'Internal server error',
                'details': str(e),
                'river_available': RIVER_AVAILABLE
            })
        }
