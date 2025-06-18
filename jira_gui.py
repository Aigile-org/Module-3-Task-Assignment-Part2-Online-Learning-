import tkinter as tk
from tkinter import ttk, scrolledtext
import requests
import json
from datetime import datetime
import os
import threading
import queue
from parametars import TRAIN

# ==============================================================================
#  1. THE CORE JIRA LOGIC (Slightly modified to communicate with the GUI)
# ==============================================================================
# This is the script we've built, now wrapped in a function.
# Instead of printing, it puts messages into a queue for the GUI to display.

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

def fetch_and_transform_issues(config, output_queue):
    """Fetches and transforms Jira issues, putting status and results into a queue."""
    try:
        # --- Configuration from GUI ---
        JIRA_URL = config['url']
        JIRA_EMAIL = config['email']
        API_TOKEN = config['token']
        PROJECT_KEY = config['project']

        output_queue.put(f"Fetching issues from project: {PROJECT_KEY}...")

        search_url = f"{JIRA_URL}/rest/api/3/search"
        auth = (JIRA_EMAIL, API_TOKEN)
        headers = {"Accept": "application/json"}
        jql_query = f'project = "{PROJECT_KEY}"'
        requested_fields = [
            'summary', 'description', 'status', 'issuetype', 'reporter', 'assignee',
            'components', 'created', 'labels', 'priority', 'project'
        ]
        
        all_issues_raw = []
        start_at = 0
        while True:
            params = {
                'jql': jql_query,
                'startAt': start_at,
                'maxResults': 100,
                'fields': ','.join(requested_fields)
            }
            response = requests.get(search_url, headers=headers, params=params, auth=auth, timeout=30)
            response.raise_for_status()
            data = response.json()
            issues_on_page = data.get('issues', [])
            all_issues_raw.extend(issues_on_page)
            total_issues = data.get('total', 0)
            
            output_queue.put(f"Fetched {len(all_issues_raw)} of {total_issues} issues...")
            
            if len(all_issues_raw) >= total_issues:
                break
            start_at += len(issues_on_page)

        output_queue.put("Transforming data into the desired structure...")
        
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
                "_id": issue.get('id'), "assignee": assignee_name, "components": fields.get('components', []),
                "created": formatted_created_date, "description": get_text_from_description(fields.get('description')),
                "issuetype": fields.get('issuetype'), "labels": fields.get('labels', []),
                "priority": fields.get('priority'), "projectname": fields.get('project', {}).get('name'),
                "summary": fields.get('summary')
            }
            transformed_issues_list.append(transformed_issue)

        # Save to file
        output_folder = r"data"
        output_filename = f"{PROJECT_KEY}.json"
        output_path = os.path.join(output_folder, output_filename)
        os.makedirs(output_folder, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(transformed_issues_list, f, indent=2)

        output_queue.put("\n--- SUCCESS ---")
        output_queue.put(f"Transformed issue data saved to: {output_path}")
        output_queue.put("\n--- JSON OUTPUT ---")
        output_queue.put(json.dumps(transformed_issues_list, indent=2))

    except Exception as e:
        output_queue.put(f"\n--- ERROR ---")
        output_queue.put(f"An error occurred: {e}")
    finally:
        output_queue.put("TASK_COMPLETE")


def Test_fetch_and_transform_issues(config, output_queue):
    """Fetches and transforms Jira issues, saving each to a separate file."""
    try:
        # --- Configuration from GUI ---
        JIRA_URL = config['url']
        JIRA_EMAIL = config['email']
        API_TOKEN = config['token']
        PROJECT_KEY = config['project']

        output_queue.put(f"Fetching issues from project: {PROJECT_KEY}...")

        search_url = f"{JIRA_URL}/rest/api/3/search"
        auth = (JIRA_EMAIL, API_TOKEN)
        headers = {"Accept": "application/json"}
        jql_query = f'project = "{PROJECT_KEY}"'
        requested_fields = [
            'summary', 'description', 'status', 'issuetype', 'reporter', 'assignee',
            'components', 'created', 'labels', 'priority', 'project'
        ]
        
        all_issues_raw = []
        start_at = 0
        while True:
            params = {
                'jql': jql_query,
                'startAt': start_at,
                'maxResults': 100,
                'fields': ','.join(requested_fields)
            }
            response = requests.get(search_url, headers=headers, params=params, auth=auth, timeout=30)
            response.raise_for_status()
            data = response.json()
            issues_on_page = data.get('issues', [])
            all_issues_raw.extend(issues_on_page)
            total_issues = data.get('total', 0)
            
            output_queue.put(f"Fetched {len(all_issues_raw)} of {total_issues} issues...")
            
            if len(all_issues_raw) >= total_issues:
                break
            start_at += len(issues_on_page)

        output_queue.put("Transforming data and saving individual files...")
        
        # --- Create the output directory once ---
        output_folder = r"deploy and test\jsons"
        os.makedirs(output_folder, exist_ok=True)
        
        transformed_issues_list = []
        # --- Loop, transform, AND save each issue ---
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
                "_id": issue.get('id'),
                "key": issue.get('key'), # Also good to have the key in the transformed data
                "assignee": assignee_name, 
                "components": fields.get('components', []),
                "created": formatted_created_date, 
                "description": get_text_from_description(fields.get('description')),
                "issuetype": fields.get('issuetype'), 
                "labels": fields.get('labels', []),
                "priority": fields.get('priority'), 
                "projectname": fields.get('project', {}).get('name'),
                "summary": fields.get('summary')
            }
            transformed_issues_list.append(transformed_issue)
            
            # --- Save this single issue to its own file ---
            issue_key = issue.get('key')
            if issue_key:
                # Sanitize key for filename if necessary, but Jira keys are usually safe
                output_filename = f"{issue_key}.json"
                output_path = os.path.join(output_folder, output_filename)
                with open(output_path, 'w') as f:
                    json.dump(transformed_issue, f, indent=2)

        output_queue.put("\n--- SUCCESS ---")
        output_queue.put(f"{len(transformed_issues_list)} issues saved as individual files in: {output_folder}")
        output_queue.put("\n--- COMBINED JSON OUTPUT (for display) ---")
        output_queue.put(json.dumps(transformed_issues_list, indent=2))

    except Exception as e:
        output_queue.put(f"\n--- ERROR ---")
        output_queue.put(f"An error occurred: {e}")
    finally:
        output_queue.put("TASK_COMPLETE")

# ==============================================================================
#  2. THE GUI APPLICATION CLASS
# ==============================================================================

class JiraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Jira Issue Fetcher")
        self.root.geometry("700x600")

        # --- Create a queue for thread communication ---
        self.queue = queue.Queue()

        # --- Main Frame ---
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Input Frame ---
        input_frame = ttk.LabelFrame(main_frame, text="Jira Configuration", padding="10")
        input_frame.pack(fill=tk.X, expand=True)

        # --- Input Fields ---
        self.entries = {}
        fields_to_create = {
            "JIRA_URL": ("Jira URL:", "https://mdr21.atlassian.net/", False),
            "JIRA_EMAIL": ("Jira Email:", "abuyehia100@gmail.com", False),
            "API_TOKEN": ("API Token:", "ATATT3xFfGF09UtsyqZ7HzlPDKVPU3qVUnkVL3sss6I6XaYGWf10Mg8y3QSlJ9XlDTeGLf_V54_RZugWewygmR2lW1U_Dm8AzFUGweywHvbV8Yk04eey8iM5YXE3unfwNtT_qbSiJuYOZg-nQnMmFmVjpYM-VyXr7wk4Bmmjd-gbd3dqE9nMFss=7028947F", True),
            "PROJECT_KEY": ("Project Name/Key:", "MY Scrum Project", False)
        }
        
        for i, (key, (label_text, default_value, is_secret)) in enumerate(fields_to_create.items()):
            label = ttk.Label(input_frame, text=label_text)
            label.grid(row=i, column=0, sticky="w", padx=5, pady=5)
            
            show_char = "*" if is_secret else ""
            entry = ttk.Entry(input_frame, width=60, show=show_char)
            entry.grid(row=i, column=1, sticky="ew", padx=5, pady=5)
            entry.insert(0, default_value)
            self.entries[key] = entry

        input_frame.columnconfigure(1, weight=1)

        # --- Action Button ---
        self.fetch_button = ttk.Button(main_frame, text="Fetch Issues", command=self.start_fetch_thread)
        self.fetch_button.pack(pady=10, fill=tk.X)

        # --- Output Area ---
        output_frame = ttk.LabelFrame(main_frame, text="Output Log", padding="10")
        output_frame.pack(fill=tk.BOTH, expand=True)
        
        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, state='disabled')
        self.output_text.pack(fill=tk.BOTH, expand=True)

        # --- Start polling the queue ---
        self.process_queue()

    def start_fetch_thread(self):
        self.fetch_button.config(state='disabled')
        self.output_text.config(state='normal')
        self.output_text.delete(1.0, tk.END)
        self.output_text.config(state='disabled')
        
        config = {
            'url': self.entries['JIRA_URL'].get(),
            'email': self.entries['JIRA_EMAIL'].get(),
            'token': self.entries['API_TOKEN'].get(),
            'project': self.entries['PROJECT_KEY'].get()
        }
        
        # Run the long task in a separate thread
        if TRAIN:
            self.thread = threading.Thread(target=fetch_and_transform_issues, args=(config, self.queue))
        else:
            self.thread = threading.Thread(target=Test_fetch_and_transform_issues, args=(config, self.queue))
        self.thread.start()

    def process_queue(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                if msg == "TASK_COMPLETE":
                    self.fetch_button.config(state='normal')
                else:
                    self.output_text.config(state='normal')
                    self.output_text.insert(tk.END, msg + "\n")
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
    root = tk.Tk()
    app = JiraApp(root)
    root.mainloop()