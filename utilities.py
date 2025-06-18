import json
import pandas as pd
import os
import pickle 
from collections import defaultdict
from itertools import repeat
import random

DATA_FOLDER = "data/"
RESULTS_FOLDER = "results_new_1/"
MODEL_SAVE_FOLDER = "saved_models/"


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
    final_accuracy = model.running_accuracy.get()
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
    
    
    # save_model(model, project_name)
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
    from running_models import MySuperEnhancedModel

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
            
        print(f"Model pre-training complete. Final accuracy on training data: {model.running_accuracy.get():.4f}")
        
        # Save the trained model for future use
        save_model(model, training_project)
    else:
        print("\nUsing pre-trained model. Skipping training phase.")
    
    print("\nThe model is now 'live' and ready for new issues.")
    print(f"Current model accuracy: {model.running_accuracy.get():.2%}\n")

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
        current_accuracy = model.running_accuracy.get()
        print(f"\n  Feedback received. Model has learned from assigning the issue to '{true_label}'.")
        print(f"  Updated model accuracy: {current_accuracy:.2%}")

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
