import tkinter as tk
from tkinter import ttk, scrolledtext
import os
import json
import pandas as pd
import threading
import queue

# ==============================================================================
#  1. THE CORE PROCESSING LOGIC (Wrapped in a function)
# ==============================================================================
# This is your script, now as a function that takes the mode ('train' or 'test')
# and communicates with the GUI through a queue.

def process_json_to_csv(mode, output_queue):
    """
    Reads Jira issue JSON files, processes them into a structured format,
    and saves the result as a CSV file.
    """
    try:
        # --- Configuration based on mode ---
        if mode == 'train':
            DATA_FOLDER = "data/"
            output_queue.put("--- Running in TRAIN mode ---")
        else: # test mode
            DATA_FOLDER = r"deploy and test\jsons"
            output_queue.put("--- Running in TEST mode ---")

        # Step 2: Find all .json files in the data folder
        output_queue.put(f"Searching for .json files in: {DATA_FOLDER}")
        try:
            json_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".json") and "name_mapping" not in f]
            if not json_files:
                output_queue.put("No .json files found in the directory.")
        except FileNotFoundError:
            output_queue.put(f"Error: The directory '{DATA_FOLDER}' was not found.")
            output_queue.put("Please make sure the folder exists and the path is correct.")
            json_files = [] # Ensure the list is empty

        # Loop over each file found
        for project_file_name in json_files:
            output_queue.put(f"\n--- Processing project file: {project_file_name} ---")

            file_path = os.path.join(DATA_FOLDER, project_file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                # Handle both single JSON object (for test files) and list of objects (for train file)
                raw_data = json.load(f)
                if isinstance(raw_data, dict):
                    issues_data = [raw_data] # If it's a single issue, put it in a list
                else:
                    issues_data = raw_data # If it's already a list, use it as is

            processed_issues = []
            
            # Loop through each issue in the loaded data
            for raw_issue in issues_data:
                # *** CORRECTED INDENTATION STARTS HERE ***
                clean_issue = {}
                clean_issue["id"] = raw_issue.get("_id") # Use .get() for safety
                clean_issue["summary"] = raw_issue.get("summary")
                clean_issue["description"] = raw_issue.get("description", "")
                clean_issue["project_name"] = raw_issue.get("projectname")
                clean_issue["created_date"] = raw_issue.get("created")
                clean_issue["assignee"] = raw_issue.get("assignee")
                
                if raw_issue.get("issuetype"):
                    clean_issue["issue_type_name"] = raw_issue["issuetype"].get("name")
                if raw_issue.get("priority"):
                    clean_issue["priority_name"] = raw_issue["priority"].get("name")
                
                clean_issue["labels"] = " ".join(raw_issue.get("labels", []))
                
                component_names = [comp.get("name") for comp in raw_issue.get("components", [])]
                clean_issue["components_name"] = component_names
                
                processed_issues.append(clean_issue)
                # *** CORRECTED INDENTATION ENDS HERE ***

            if not processed_issues:
                output_queue.put(f"No processable issues found in {project_file_name}. Skipping.")
                continue

            df = pd.DataFrame(processed_issues)
            df["assignee_num"], name_mapping_array = pd.factorize(df["assignee"])
            
            # --- Save name mapping and final CSV ---
            # Use project name from the first row for consistent naming
            project_name = df["project_name"][0] if not df.empty and "project_name" in df.columns else "unknown_project"

            name_mapping_series = pd.Series(name_mapping_array.to_list(), name="assignee_name")
            mapping_filename = os.path.join(DATA_FOLDER, f"{project_name}_name_mapping.csv")
            name_mapping_series.to_csv(mapping_filename, index=False)
            output_queue.put(f"Saved assignee name mapping to: {mapping_filename}")

            if mode == 'train':
                output_csv_folder = "data/"
                output_filename = os.path.join(output_csv_folder, f"{project_name}.csv")
            else: # test mode
                output_csv_folder = r"deploy and test\csvs"
                output_filename = os.path.join(output_csv_folder, f"{project_name}_test.csv")
            
            os.makedirs(output_csv_folder, exist_ok=True)
            df.to_csv(output_filename, index=False)
            output_queue.put(f"Successfully processed {len(df)} issues.")
            output_queue.put(f"Output saved to: {output_filename}")
        
        output_queue.put("\n--- All files processed. ---")

    except Exception as e:
        output_queue.put(f"\n--- AN UNEXPECTED ERROR OCCURRED ---")
        output_queue.put(f"Error: {e}")
    finally:
        output_queue.put("TASK_COMPLETE")

# ==============================================================================
#  2. THE GUI APPLICATION CLASS
# ==============================================================================

class DataProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Jira Data Processor")
        self.root.geometry("700x500")

        self.queue = queue.Queue()

        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Action Frame ---
        action_frame = ttk.LabelFrame(main_frame, text="Select Mode", padding="10")
        action_frame.pack(fill=tk.X)
        
        self.train_button = ttk.Button(action_frame, text="Process Train Data", command=lambda: self.start_processing_thread('train'))
        self.train_button.pack(side=tk.LEFT, padx=5, pady=5, expand=True, fill=tk.X)
        
        self.test_button = ttk.Button(action_frame, text="Process Test Data", command=lambda: self.start_processing_thread('test'))
        self.test_button.pack(side=tk.LEFT, padx=5, pady=5, expand=True, fill=tk.X)
        
        # --- Output Log Area ---
        output_frame = ttk.LabelFrame(main_frame, text="Output Log", padding="10")
        output_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, state='disabled')
        self.output_text.pack(fill=tk.BOTH, expand=True)

        self.process_queue()

    def start_processing_thread(self, mode):
        # Disable buttons to prevent multiple clicks
        self.train_button.config(state='disabled')
        self.test_button.config(state='disabled')
        
        # Clear previous logs
        self.output_text.config(state='normal')
        self.output_text.delete(1.0, tk.END)
        self.output_text.config(state='disabled')
        
        # Start the processing in a background thread
        self.thread = threading.Thread(target=process_json_to_csv, args=(mode, self.queue))
        self.thread.start()

    def process_queue(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                if msg == "TASK_COMPLETE":
                    # Re-enable buttons
                    self.train_button.config(state='normal')
                    self.test_button.config(state='normal')
                else:
                    self.output_text.config(state='normal')
                    self.output_text.insert(tk.END, str(msg) + "\n")
                    self.output_text.see(tk.END) # Scroll to the bottom
                    self.output_text.config(state='disabled')
        except queue.Empty:
            pass # No message in queue
        
        # Check again after 100ms
        self.root.after(100, self.process_queue)

# ==============================================================================
#  3. BOILERPLATE TO RUN THE APP
# ==============================================================================
if __name__ == "__main__":
    # Remove the import from `parametars` as the GUI now controls the mode
    # from parametars import TEST 
    
    root = tk.Tk()
    app = DataProcessorApp(root)
    root.mainloop()