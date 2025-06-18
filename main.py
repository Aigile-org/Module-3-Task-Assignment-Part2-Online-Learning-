from running_models import MyNaiveBayesModel, MyNaiveBayesWithADWIN, MyEnhancedModel, HoeffdingAdaptiveTreeModel,PassiveAggressiveModel,LeveragingBaggingModel, MySuperEnhancedModel
from utilities import run_single_experiment, run_deployment_simulation, print_summary_table, DATA_FOLDER
import os
import multiprocessing as mp

# ==============================================================================
# Part 4: Main Execution Block
# ==============================================================================
if __name__ == '__main__':

    # --- CHOOSE YOUR MODE ---
    # Set to 'experiment' to run the full parallel simulations.
    # Set to 'deploy' to run the interactive deployment simulation.
    MODE = 'e' 

    if MODE == 'e':
        print("Script execution started in EXPERIMENT mode.")
        # --- Your existing code for running parallel experiments ---
        models_to_test = [PassiveAggressiveModel] # etc.
        projects_to_process = [f[:-4] for f in os.listdir(DATA_FOLDER) if f.endswith(".csv")]
        if projects_to_process:
            experiment_tasks = [(m, p) for m in models_to_test for p in projects_to_process]
            num_workers = min(mp.cpu_count(), 4)
            print(f"Starting experiments in PARALLEL using {num_workers} worker processes...")
            with mp.Pool(processes=num_workers) as pool:
                results_list = pool.starmap(run_single_experiment, experiment_tasks)
            print("\nAll experiments have finished.")
            summary_data = {}
            for task, final_accuracy in zip(experiment_tasks, results_list):
                model_class, project_name = task
                model_name = model_class().name
                if project_name not in summary_data:
                    summary_data[project_name] = {}
                summary_data[project_name][model_name] = final_accuracy
            print_summary_table(summary_data, models_to_test)
        else:
            print("No projects found to process. Halting execution.")

    elif MODE == 'd':
        print("Script execution started in DEPLOYMENT SIMULATION mode.")
        
        # --- Configuration for the deployment simulation ---
        PROJECT_FOR_TRAINING = "DATALAB" 
        DEPLOYMENT_TEST_FILE = "DEPLOYMENT_TEST.csv"
        test_file_path = os.path.join(DATA_FOLDER, DEPLOYMENT_TEST_FILE)
        
        # Run the interactive simulation
        run_deployment_simulation(
            training_project=PROJECT_FOR_TRAINING,
            test_file_path=test_file_path,
            use_pretrained=True  # Set to False to force retraining
        )

    else:
        print(f"Error: Unknown mode '{MODE}'. Please choose 'experiment' or 'deploy'.")