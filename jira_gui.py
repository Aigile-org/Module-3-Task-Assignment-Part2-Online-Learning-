import tkinter as tk
from tkinter import ttk, scrolledtext
import requests
import json
from datetime import datetime
import os
import threading
import queue

# ==============================================================================
#  1. CORE JIRA LOGIC (Two separate functions for Train and Test modes)
# ==============================================================================

def get_text_from_description(description_obj):
    if not description_obj or 'content' not in description_obj:
        return ""
    text_parts = []
    for content_block in description_obj['content']:
        if 'content' in content_block:
            for text_item in content_block['content']:
                if 'text' in text_item:
                    text_parts.append(text_item['text'])
    return "\n".join(text_parts)

# --- TRAIN MODE FUNCTION ---
# (Fetches all issues from a project and saves to a single file)
def fetch_and_transform_issues(config, output_queue):
    """TRAIN MODE: Fetches and transforms Jira issues, saving all to one file."""
    try:
        JIRA_URL, JIRA_EMAIL, API_TOKEN, PROJECT_KEY = config['url'], config['email'], config['token'], config['project']
        output_queue.put(f"TRAIN MODE: Fetching all issues from project: {PROJECT_KEY}...")
        
        search_url = f"{JIRA_URL}/rest/api/3/search"
        auth = (JIRA_EMAIL, API_TOKEN)
        headers = {"Accept": "application/json"}
        jql_query = f'project = "{PROJECT_KEY}"'
        requested_fields = ['summary', 'description', 'status', 'issuetype', 'reporter', 'assignee', 'components', 'created', 'labels', 'priority', 'project']
        all_issues_raw = []
        start_at = 0
        while True:
            params = {'jql': jql_query, 'startAt': start_at, 'maxResults': 100, 'fields': ','.join(requested_fields)}
            response = requests.get(search_url, headers=headers, params=params, auth=auth, timeout=30)
            response.raise_for_status()
            data = response.json()
            issues_on_page = data.get('issues', [])
            all_issues_raw.extend(issues_on_page)
            total_issues = data.get('total', 0)
            output_queue.put(f"Fetched {len(all_issues_raw)} of {total_issues} issues...")
            if len(all_issues_raw) >= total_issues: break
            start_at += len(issues_on_page)

        output_queue.put("Transforming data...")
        transformed_issues_list = []
        for issue in all_issues_raw:
            fields = issue.get('fields', {})
            assignee_obj = fields.get('assignee')
            assignee_name = assignee_obj['displayName'] if assignee_obj else None
            created_date_str = fields.get('created')
            formatted_created_date = None
            if created_date_str:
                created_datetime = datetime.strptime(created_date_str, "%Y-%m-%dT%H:%M:%S.%f%z")
                formatted_created_date = created_datetime.strftime("%Y-%m-%d %H:%M:%S")
            transformed_issue = {
                "_id": issue.get('id'), "key": issue.get('key'), "assignee": assignee_name, "components": fields.get('components', []),
                "created": formatted_created_date, "description": get_text_from_description(fields.get('description')),
                "issuetype": fields.get('issuetype'), "labels": fields.get('labels', []),
                "priority": fields.get('priority'), "projectname": fields.get('project', {}).get('name'),
                "summary": fields.get('summary')
            }
            transformed_issues_list.append(transformed_issue)
        
        output_folder = r"data"
        output_filename = f"{PROJECT_KEY}.json"
        output_path = os.path.join(output_folder, output_filename)
        os.makedirs(output_folder, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(transformed_issues_list, f, indent=2)

        output_queue.put(f"\n--- SUCCESS (TRAIN MODE) ---\nAll {len(transformed_issues_list)} issues saved to: {output_path}")
        output_queue.put("\n--- JSON OUTPUT ---\n" + json.dumps(transformed_issues_list, indent=2))
    except Exception as e:
        output_queue.put(f"\n--- ERROR ---\nAn error occurred: {e}")
    finally:
        output_queue.put("TASK_COMPLETE")


# --- TEST MODE FUNCTION (MODIFIED) ---
# (Fetches a single issue by URL and saves it to a single file)
def Test_fetch_and_transform_issues(config, output_queue):
    """TEST MODE: Fetches a single Jira issue by URL and saves it."""
    try:
        JIRA_URL, JIRA_EMAIL, API_TOKEN, ISSUE_URL = config['url'], config['email'], config['token'], config['issue_url']
        
        if not ISSUE_URL or '/' not in ISSUE_URL:
            raise ValueError("Invalid Issue URL provided.")
            
        # Extract issue key from URL (e.g., "PROJ-123")
        issue_key = ISSUE_URL.strip().split('/')[-1]
        output_queue.put(f"TEST MODE: Fetching single issue: {issue_key}...")
        
        issue_api_url = f"{JIRA_URL}/rest/api/3/issue/{issue_key}"
        auth = (JIRA_EMAIL, API_TOKEN)
        headers = {"Accept": "application/json"}
        requested_fields = ['summary', 'description', 'status', 'issuetype', 'reporter', 'assignee', 'components', 'created', 'labels', 'priority', 'project']
        params = {'fields': ','.join(requested_fields)}

        response = requests.get(issue_api_url, headers=headers, params=params, auth=auth, timeout=30)
        response.raise_for_status() # Will raise an error for 404 Not Found, etc.
        
        # The response for a single issue is the issue object itself, not a list
        issue_raw = response.json()
        
        output_queue.put("Transforming data and saving file...")

        # Transform the single issue
        fields = issue_raw.get('fields', {})
        assignee_obj = fields.get('assignee')
        assignee_name = assignee_obj['displayName'] if assignee_obj else None
        created_date_str = fields.get('created')
        formatted_created_date = None
        if created_date_str:
            created_datetime = datetime.strptime(created_date_str, "%Y-%m-%dT%H:%M:%S.%f%z")
            formatted_created_date = created_datetime.strftime("%Y-%m-%d %H:%M:%S")
        
        transformed_issue = {
            "_id": issue_raw.get('id'), "key": issue_raw.get('key'), "assignee": assignee_name, "components": fields.get('components', []),
            "created": formatted_created_date, "description": get_text_from_description(fields.get('description')),
            "issuetype": fields.get('issuetype'), "labels": fields.get('labels', []),
            "priority": fields.get('priority'), "projectname": fields.get('project', {}).get('name'),
            "summary": fields.get('summary')
        }

        # Save the single issue to its own file
        output_folder = r"deploy and test\jsons"
        os.makedirs(output_folder, exist_ok=True)
        output_filename = f"{issue_key}.json"
        output_path = os.path.join(output_folder, output_filename)
        with open(output_path, 'w') as f:
            json.dump(transformed_issue, f, indent=2)

        output_queue.put(f"\n--- SUCCESS (TEST MODE) ---\nIssue {issue_key} saved to: {output_path}")
        output_queue.put("\n--- JSON OUTPUT ---\n" + json.dumps(transformed_issue, indent=2))

    except Exception as e:
        output_queue.put(f"\n--- ERROR ---\nAn error occurred: {e}")
    finally:
        output_queue.put("TASK_COMPLETE")


# ==============================================================================
#  2. THE MAIN GUI APPLICATION CLASS (MODIFIED)
# ==============================================================================

class JiraApp:
    def __init__(self, root, mode):
        self.root = root
        self.mode = mode # 'train' or 'test'
        self.root.title(f"Jira Issue Fetcher ({self.mode.capitalize()} Mode)")
        self.root.geometry("700x600")
        self.queue = queue.Queue()
        
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        input_frame = ttk.LabelFrame(main_frame, text="Jira Configuration", padding="10")
        input_frame.pack(fill=tk.X)

        # --- Create all possible input fields ---
        self.widgets = {}
        fields_to_create = {
            "JIRA_URL": ("Jira URL:", "https://mdr21.atlassian.net/", False),
            "JIRA_EMAIL": ("Jira Email:", "abuyehia100@gmail.com", False),
            "API_TOKEN": ("API Token:", "ATATT3xFfGF09UtsyqZ7HzlPDKVPU3qVUnkVL3sss6I6XaYGWf10Mg8y3QSlJ9XlDTeGLf_V54_RZugWewygmR2lW1U_Dm8AzFUGweywHvbV8Yk04eey8iM5YXE3unfwNtT_qbSiJuYOZg-nQnMmFmVjpYM-VyXr7wk4Bmmjd-gbd3dqE9nMFss=7028947F", True),
            "PROJECT_KEY": ("Project Name/Key:", "MY Scrum Project", False),
            "ISSUE_URL": ("Issue URL:", "https://mdr21.atlassian.net/browse/SCRUM-3", False)
        }
        for i, (key, (label_text, default_value, is_secret)) in enumerate(fields_to_create.items()):
            label = ttk.Label(input_frame, text=label_text)
            label.grid(row=i, column=0, sticky="w", padx=5, pady=5)
            show_char = "*" if is_secret else ""
            entry = ttk.Entry(input_frame, width=60, show=show_char)
            entry.grid(row=i, column=1, sticky="ew", padx=5, pady=5)
            entry.insert(0, default_value)
            self.widgets[key] = {'label': label, 'entry': entry}
        input_frame.columnconfigure(1, weight=1)

        # --- Show/Hide widgets based on mode ---
        if self.mode == 'train':
            self.widgets['ISSUE_URL']['label'].grid_remove()
            self.widgets['ISSUE_URL']['entry'].grid_remove()
        else: # 'test' mode
            self.widgets['PROJECT_KEY']['label'].grid_remove()
            self.widgets['PROJECT_KEY']['entry'].grid_remove()

        # --- Action Button and Output Area ---
        self.fetch_button = ttk.Button(main_frame, text="Fetch Issues", command=self.start_fetch_thread)
        self.fetch_button.pack(pady=10, fill=tk.X)
        output_frame = ttk.LabelFrame(main_frame, text="Output Log", padding="10")
        output_frame.pack(fill=tk.BOTH, expand=True)
        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, state='disabled')
        self.output_text.pack(fill=tk.BOTH, expand=True)
        self.process_queue()

    def start_fetch_thread(self):
        self.fetch_button.config(state='disabled')
        self.output_text.config(state='normal'); self.output_text.delete(1.0, tk.END); self.output_text.config(state='disabled')
        
        # --- Get config based on mode ---
        config = {
            'url': self.widgets['JIRA_URL']['entry'].get(), 
            'email': self.widgets['JIRA_EMAIL']['entry'].get(),
            'token': self.widgets['API_TOKEN']['entry'].get()
        }
        if self.mode == 'train':
            config['project'] = self.widgets['PROJECT_KEY']['entry'].get()
            target_function = fetch_and_transform_issues
        else: # test mode
            config['issue_url'] = self.widgets['ISSUE_URL']['entry'].get()
            target_function = Test_fetch_and_transform_issues
        
        self.thread = threading.Thread(target=target_function, args=(config, self.queue))
        self.thread.start()

    def process_queue(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                if msg == "TASK_COMPLETE": self.fetch_button.config(state='normal')
                else:
                    self.output_text.config(state='normal')
                    self.output_text.insert(tk.END, str(msg) + "\n")
                    self.output_text.see(tk.END)
                    self.output_text.config(state='disabled')
        except queue.Empty: pass
        self.root.after(100, self.process_queue)


# ==============================================================================
#  3. THE INITIAL MODE SELECTOR CLASS
# ==============================================================================

class ModeSelector:
    def __init__(self, root):
        self.root = root
        self.root.title("Select Mode")
        self.root.geometry("300x120")
        frame = ttk.Frame(root, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        label = ttk.Label(frame, text="Please select the execution mode:")
        label.pack(pady=5)
        train_button = ttk.Button(frame, text="Run in Train Mode", command=lambda: self.launch_main_app("train"))
        train_button.pack(pady=5, fill=tk.X)
        test_button = ttk.Button(frame, text="Run in Test Mode", command=lambda: self.launch_main_app("test"))
        test_button.pack(pady=5, fill=tk.X)

    def launch_main_app(self, mode):
        self.root.destroy()
        main_root = tk.Tk()
        JiraApp(main_root, mode=mode)
        main_root.mainloop()

# ==============================================================================
#  4. BOILERPLATE TO RUN THE APP
# ==============================================================================
if __name__ == "__main__":
    selector_root = tk.Tk()
    app_selector = ModeSelector(selector_root)
    selector_root.mainloop()