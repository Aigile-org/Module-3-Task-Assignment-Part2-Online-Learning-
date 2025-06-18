import tkinter as tk
from tkinter import ttk, filedialog
import subprocess
import sys
import os

class ModelLauncherGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ML Model Launcher")
        self.root.geometry("550x300")

        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Create Tabs for Different Modes ---
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=5)

        # --- Create the individual tabs ---
        self.create_deploy_tab()
        self.create_experiment_tab()
        
        # --- Status Bar ---
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_deploy_tab(self):
        deploy_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(deploy_frame, text="Deployment Simulation")

        # --- Input fields for deployment mode ---
        ttk.Label(deploy_frame, text="Training Project Name:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.train_project_entry = ttk.Entry(deploy_frame, width=40)
        self.train_project_entry.grid(row=0, column=1, sticky=tk.EW, pady=5)
        self.train_project_entry.insert(0, "AMBARI")

        ttk.Label(deploy_frame, text="Deployment Test File:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.test_file_entry = ttk.Entry(deploy_frame, width=40)
        self.test_file_entry.grid(row=1, column=1, sticky=tk.EW, pady=5)
        self.test_file_entry.insert(0, "My Scrum Project_test.csv")
        
        browse_button = ttk.Button(deploy_frame, text="Browse...", command=self.browse_test_file)
        browse_button.grid(row=1, column=2, padx=5)

        self.force_retrain_var = tk.BooleanVar()
        ttk.Checkbutton(
            deploy_frame, 
            text="Force Retraining (don't use saved model)", 
            variable=self.force_retrain_var
        ).grid(row=2, column=1, sticky=tk.W, pady=10)

        # --- Launch Button ---
        run_deploy_button = ttk.Button(deploy_frame, text="Run Deployment Simulation", command=self.run_deployment)
        run_deploy_button.grid(row=3, column=0, columnspan=3, pady=10, sticky=tk.EW)

        deploy_frame.columnconfigure(1, weight=1)
        
    def create_experiment_tab(self):
        exp_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(exp_frame, text="Run Experiments")

        ttk.Label(
            exp_frame, 
            text="This mode will run all defined models on all project CSVs found in the 'data' folder.\n\nThis may take a long time and use significant CPU resources.",
            wraplength=400,
            justify=tk.LEFT
        ).pack(pady=10)

        run_exp_button = ttk.Button(exp_frame, text="Run All Experiments", command=self.run_experiments)
        run_exp_button.pack(pady=20, fill=tk.X)

    def browse_test_file(self):
        # We assume the test files are in the 'deploy and test\csvs' directory
        initial_dir = "deploy and test/csvs"
        if not os.path.isdir(initial_dir):
            initial_dir = os.getcwd() # Fallback to current directory
            
        filepath = filedialog.askopenfilename(
            initialdir=initial_dir,
            title="Select Test File",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if filepath:
            # We only need the filename, not the whole path
            filename = os.path.basename(filepath)
            self.test_file_entry.delete(0, tk.END)
            self.test_file_entry.insert(0, filename)

    def run_deployment(self):
        train_project = self.train_project_entry.get()
        test_file = self.test_file_entry.get()

        if not train_project or not test_file:
            self.status_var.set("Error: All fields are required for deployment mode.")
            return

        command = [
            sys.executable,  # The current python interpreter
            "running_models.py",
            "--mode", "deploy",
            "--train_project", train_project,
            "--test_file", test_file
        ]
        
        if self.force_retrain_var.get():
            command.append("--no_pretrained")

        self.launch_script(command)

    def run_experiments(self):
        command = [
            sys.executable,
            "running_models.py",
            "--mode", "experiment"
        ]
        self.launch_script(command)

    def launch_script(self, command):
        """Launches the command in a new terminal window."""
        try:
            # This logic opens a new terminal window to run the script.
            # It's different for Windows vs. Mac/Linux.
            if os.name == 'nt': # Windows
                subprocess.Popen(['cmd.exe', '/c', 'start', 'cmd', '/k'] + command)
            elif sys.platform == 'darwin': # macOS
                # This requires running the GUI from a terminal to work
                subprocess.Popen(['open', '-a', 'Terminal.app'] + [sys.executable] + command[1:])
            else: # Linux
                # Tries to open in gnome-terminal, falls back to xterm
                try:
                    subprocess.Popen(['gnome-terminal', '--'] + command)
                except FileNotFoundError:
                    subprocess.Popen(['xterm', '-e'] + command)
            
            self.status_var.set(f"Launched '{command[2]}' in a new terminal window.")
        except Exception as e:
            self.status_var.set(f"Error launching script: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ModelLauncherGUI(root)
    root.mainloop()