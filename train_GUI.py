import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import requests
import json
from datetime import datetime
import os
import threading
import queue
import pickle
import pandas as pd

# AWS S3 imports
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

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
# (Fetches all issues from a project, trains model, and saves to S3)
def fetch_train_and_upload(config, output_queue):
    """TRAIN MODE: Fetches issues, trains model, and uploads to S3."""
    try:
        JIRA_URL, JIRA_EMAIL, API_TOKEN, PROJECT_KEY = config['url'], config['email'], config['token'], config['project']
        S3_BUCKET = config.get('s3_bucket', 'aigile-bucket')
        S3_KEY = config.get('s3_key', 'My_Super_Enhanced_Model_AMBARI.pkl')
        
        output_queue.put(f"🔄 TRAIN MODE: Fetching all issues from project: {PROJECT_KEY}...")
        
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
            output_queue.put(f"📥 Fetched {len(all_issues_raw)} of {total_issues} issues...")
            if len(all_issues_raw) >= total_issues: break
            start_at += len(issues_on_page)

        output_queue.put("🔄 Transforming data...")
        transformed_issues_list = []
        assignees_found = set()
        name_mapping = []
        assignee_to_num = {}
        
        for issue in all_issues_raw:
            fields = issue.get('fields', {})
            assignee_obj = fields.get('assignee')
            assignee_name = assignee_obj['displayName'] if assignee_obj else None
            
            if assignee_name:  # Only count issues with assignees
                assignees_found.add(assignee_name)
                # Create numeric mapping for assignees
                if assignee_name not in assignee_to_num:
                    assignee_to_num[assignee_name] = len(name_mapping)
                    name_mapping.append(assignee_name)
            
            created_date_str = fields.get('created')
            formatted_created_date = None
            if created_date_str:
                created_datetime = datetime.strptime(created_date_str, "%Y-%m-%dT%H:%M:%S.%f%z")
                formatted_created_date = created_datetime.strftime("%Y-%m-%d %H:%M:%S")
            
            # Extract components and labels as strings (like in running_models.py expects)
            components = ', '.join([comp.get('name', '') for comp in fields.get('components', [])])
            labels = ', '.join([label.get('name', label) if isinstance(label, dict) else str(label) for label in fields.get('labels', [])])
            
            # Get issue type and priority names
            issue_type = fields.get('issuetype', {})
            issue_type_name = issue_type.get('name', '') if isinstance(issue_type, dict) else str(issue_type)
            
            priority = fields.get('priority', {})
            priority_name = priority.get('name', 'Medium') if isinstance(priority, dict) else str(priority)
            
            transformed_issue = {
                "_id": issue.get('id'), 
                "key": issue.get('key'), 
                "assignee": assignee_to_num.get(assignee_name) if assignee_name else None,  # Convert to numeric ID
                "components_name": components,  # Changed from "components"
                "created_date": formatted_created_date,  # Changed from "created"
                "description": get_text_from_description(fields.get('description')),
                "issue_type_name": issue_type_name,  # Changed from "issuetype"
                "labels": labels,
                "priority_name": priority_name,  # Changed from "priority"
                "projectname": fields.get('project', {}).get('name'),
                "summary": fields.get('summary', '')
            }
            transformed_issues_list.append(transformed_issue)
        
        # Save data locally
        output_folder = r"data"
        output_filename = f"{PROJECT_KEY}.json"
        output_path = os.path.join(output_folder, output_filename)
        os.makedirs(output_folder, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(transformed_issues_list, f, indent=2)
        
        output_queue.put(f"💾 Data saved locally to: {output_path}")
        output_queue.put(f"📊 Found {len(assignees_found)} unique assignees: {', '.join(sorted(assignees_found))}")
          # Train model directly (instead of importing from running_models)
        output_queue.put("🤖 Training model...")
        try:
            # Import required classes from running_models
            from running_models import MySuperEnhancedModel, my_preprocess_data
            
            # Convert JSON data to DataFrame for processing
            df = pd.DataFrame(transformed_issues_list)
              # Preprocess the data for River
            features_stream, labels_stream = my_preprocess_data(df)
            
            output_queue.put(f"📊 Preprocessing complete - {len(features_stream)} training samples")
            
            # Create and train the model with proper name mapping
            model = MySuperEnhancedModel(name_mapping=name_mapping)
            
            # Train the model with all the data
            for i, (issue_features, true_label) in enumerate(zip(features_stream, labels_stream)):
                model.process_one_issue(
                    issue_index=i,
                    issue_features=issue_features,
                    issue_label=true_label
                )
                if (i + 1) % 10 == 0:
                    current_accuracy = model.running_accuracy.get()
                    output_queue.put(f"📈 Processed {i + 1}/{len(features_stream)} issues (accuracy: {current_accuracy:.3f})")
            
            final_accuracy = model.running_accuracy.get()
            output_queue.put(f"✅ Training complete! Final accuracy: {final_accuracy:.3f}")
            
            # Save the model locally first
            model_folder = "saved_models"
            os.makedirs(model_folder, exist_ok=True)
            model_path = os.path.join(model_folder, f"{PROJECT_KEY}_trained_model.pkl")            # Create model data dictionary like the original format
            model_data = {
                'model': model,
                'name_mapping': name_mapping
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            output_queue.put(f"✅ Model saved locally: {model_path}")
            output_queue.put(f"📊 Model file size: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")
            
            # Upload to S3 with detailed logging
            output_queue.put(f"☁️ Uploading to S3: s3://{S3_BUCKET}/{S3_KEY}")
            output_queue.put(f"🔑 Checking AWS credentials...")
            
            s3_client = boto3.client('s3')
            
            # Test connection first
            try:
                s3_client.head_bucket(Bucket=S3_BUCKET)
                output_queue.put(f"✅ S3 bucket '{S3_BUCKET}' is accessible")
            except Exception as bucket_error:
                output_queue.put(f"❌ Cannot access bucket '{S3_BUCKET}': {str(bucket_error)}")
                raise bucket_error
            
            # Upload the file
            output_queue.put(f"📤 Starting upload...")
            s3_client.upload_file(model_path, S3_BUCKET, S3_KEY)
            
            # Verify upload
            try:
                response = s3_client.head_object(Bucket=S3_BUCKET, Key=S3_KEY)
                uploaded_size = response['ContentLength']
                output_queue.put(f"✅ Upload verified! File size in S3: {uploaded_size / 1024 / 1024:.2f} MB")
                output_queue.put("✅ Model uploaded to S3 successfully!")
            except Exception as verify_error:
                output_queue.put(f"⚠️ Upload may have failed - verification error: {str(verify_error)}")
                
        except ImportError as ie:
            output_queue.put(f"❌ Could not import training script: {str(ie)}")
            output_queue.put("❌ Make sure running_models.py is available and has a main() function.")
        except NoCredentialsError:
            output_queue.put("❌ AWS credentials not found!")
            output_queue.put("💡 Please set environment variables:")
            output_queue.put("   AWS_ACCESS_KEY_ID=your-access-key")
            output_queue.put("   AWS_SECRET_ACCESS_KEY=your-secret-key") 
            output_queue.put("   AWS_DEFAULT_REGION=us-east-1")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            output_queue.put(f"❌ AWS S3 error ({error_code}): {error_message}")
            
            if error_code == 'NoSuchBucket':
                output_queue.put(f"💡 Bucket '{S3_BUCKET}' doesn't exist. Please create it in AWS Console.")
            elif error_code == 'AccessDenied':
                output_queue.put("💡 Access denied. Check your AWS permissions (s3:PutObject, s3:GetObject)")
        except Exception as e:
            output_queue.put(f"❌ Unexpected error: {str(e)}")
            import traceback
            output_queue.put(f"🐛 Full traceback: {traceback.format_exc()}")

        output_queue.put(f"\n✅ TRAINING COMPLETE!")
        output_queue.put(f"   • Fetched {len(transformed_issues_list)} issues")
        output_queue.put(f"   • Found {len([i for i in transformed_issues_list if i['assignee']])} issues with assignees")
        output_queue.put(f"   • Data saved to: {output_path}")
        output_queue.put(f"   • Model uploaded to S3: s3://{S3_BUCKET}/{S3_KEY}")
        
    except Exception as e:
        output_queue.put(f"\n❌ ERROR: {str(e)}")
    finally:
        output_queue.put("TASK_COMPLETE")


# ==============================================================================
#  2. THE MAIN GUI APPLICATION CLASS
# ==============================================================================

class JiraTrainerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Jira ML Model Trainer & S3 Uploader")
        self.root.geometry("700x600")
        self.queue = queue.Queue()
        
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Jira Configuration
        input_frame = ttk.LabelFrame(main_frame, text="Jira Configuration", padding="10")
        input_frame.pack(fill=tk.X)

        self.widgets = {}
        fields_to_create = {
            "JIRA_URL": ("Jira URL:", "https://mdr21.atlassian.net/", False),
            "JIRA_EMAIL": ("Jira Email:", "abuyehia100@gmail.com", False),
            "API_TOKEN": ("API Token:", "ATATT3xFfGF09UtsyqZ7HzlPDKVPU3qVUnkVL3sss6I6XaYGWf10Mg8y3QSlJ9XlDTeGLf_V54_RZugWewygmR2lW1U_Dm8AzFUGweywHvbV8Yk04eey8iM5YXE3unfwNtT_qbSiJuYOZg-nQnMmFmVjpYM-VyXr7wk4Bmmjd-gbd3dqE9nMFss=7028947F", True),
            "PROJECT_KEY": ("Project Name/Key:", "MY Scrum Project", False),
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

        # S3 Configuration
        s3_frame = ttk.LabelFrame(main_frame, text="S3 Configuration", padding="10")
        s3_frame.pack(fill=tk.X, pady=(10, 0))

        s3_fields = {
            "S3_BUCKET": ("S3 Bucket:", "aigile-bucket", False),
            "S3_KEY": ("S3 Model Key:", "My_Super_Enhanced_Model_AMBARI.pkl", False),
        }
        for i, (key, (label_text, default_value, is_secret)) in enumerate(s3_fields.items()):
            label = ttk.Label(s3_frame, text=label_text)
            label.grid(row=i, column=0, sticky="w", padx=5, pady=5)
            entry = ttk.Entry(s3_frame, width=60)
            entry.grid(row=i, column=1, sticky="ew", padx=5, pady=5)
            entry.insert(0, default_value)
            self.widgets[key] = {'label': label, 'entry': entry}
        s3_frame.columnconfigure(1, weight=1)

        # Action Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10, fill=tk.X)
        
        self.test_aws_button = ttk.Button(button_frame, text="🔍 Test AWS Connection", 
                                         command=self.test_aws_connection)
        self.test_aws_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.train_button = ttk.Button(button_frame, text="🚀 Fetch Data, Train Model & Upload to S3", 
                                      command=self.start_train_thread)
        self.train_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Output Area
        output_frame = ttk.LabelFrame(main_frame, text="Training Log", padding="10")
        output_frame.pack(fill=tk.BOTH, expand=True)
        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, state='disabled')
        self.output_text.pack(fill=tk.BOTH, expand=True)
        self.process_queue()

    def start_train_thread(self):
        self.train_button.config(state='disabled')
        self.output_text.config(state='normal')
        self.output_text.delete(1.0, tk.END)
        self.output_text.config(state='disabled')
        
        # Get configuration
        config = {
            'url': self.widgets['JIRA_URL']['entry'].get(), 
            'email': self.widgets['JIRA_EMAIL']['entry'].get(),
            'token': self.widgets['API_TOKEN']['entry'].get(),
            'project': self.widgets['PROJECT_KEY']['entry'].get(),
            's3_bucket': self.widgets['S3_BUCKET']['entry'].get(),
            's3_key': self.widgets['S3_KEY']['entry'].get()
        }
        
        self.thread = threading.Thread(target=fetch_train_and_upload, args=(config, self.queue))
        self.thread.start()

    def process_queue(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                if msg == "TASK_COMPLETE": 
                    self.train_button.config(state='normal')
                else:
                    self.output_text.config(state='normal')
                    self.output_text.insert(tk.END, str(msg) + "\n")
                    self.output_text.see(tk.END)
                    self.output_text.config(state='disabled')
        except queue.Empty: 
            pass
        self.root.after(100, self.process_queue)
    
    def test_aws_connection(self):
        """Test AWS S3 connection and permissions."""
        self.test_aws_button.config(state='disabled')
        try:
            bucket = self.widgets['S3_BUCKET']['entry'].get()
            key = self.widgets['S3_KEY']['entry'].get()
            
            self.add_output("🔍 Testing AWS connection...")
            self.add_output(f"   Bucket: {bucket}")
            self.add_output(f"   Key: {key}")
            
            # Test AWS credentials
            s3_client = boto3.client('s3')
            
            # Test 1: Check if bucket exists and is accessible
            self.add_output("📋 Test 1: Checking bucket access...")
            try:
                s3_client.head_bucket(Bucket=bucket)
                self.add_output(f"✅ Bucket '{bucket}' is accessible")
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == '404':
                    self.add_output(f"❌ Bucket '{bucket}' does not exist")
                elif error_code == '403':
                    self.add_output(f"❌ Access denied to bucket '{bucket}'")
                else:
                    self.add_output(f"❌ Bucket error: {error_code}")
                raise e
            
            # Test 2: Try to list objects (tests read permission)
            self.add_output("📋 Test 2: Testing read permissions...")
            response = s3_client.list_objects_v2(Bucket=bucket, MaxKeys=1)
            self.add_output("✅ Read permissions OK")
            
            # Test 3: Try to upload a test file (tests write permission)
            self.add_output("📋 Test 3: Testing write permissions...")
            test_content = "test file for AWS permissions"
            test_key = "test_aws_permissions.txt"
            
            s3_client.put_object(Bucket=bucket, Key=test_key, Body=test_content)
            self.add_output("✅ Write permissions OK")
            
            # Test 4: Verify the test file was uploaded
            self.add_output("📋 Test 4: Verifying upload...")
            response = s3_client.head_object(Bucket=bucket, Key=test_key)
            self.add_output(f"✅ Test file uploaded successfully ({response['ContentLength']} bytes)")
            
            # Clean up test file
            s3_client.delete_object(Bucket=bucket, Key=test_key)
            self.add_output("🧹 Test file cleaned up")
            
            self.add_output("🎉 AWS connection test PASSED! You're ready to upload models.")
            
        except NoCredentialsError:
            self.add_output("❌ AWS credentials not found!")
            self.add_output("💡 Set environment variables:")
            self.add_output("   AWS_ACCESS_KEY_ID=your-access-key")
            self.add_output("   AWS_SECRET_ACCESS_KEY=your-secret-key")
            self.add_output("   AWS_DEFAULT_REGION=us-east-1")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            self.add_output(f"❌ AWS error ({error_code}): {error_message}")
        except Exception as e:
            self.add_output(f"❌ Unexpected error: {str(e)}")
        finally:
            self.test_aws_button.config(state='normal')

    def add_output(self, message):
        """Helper function to add output messages to the text box."""
        self.output_text.config(state='normal')
        self.output_text.insert(tk.END, str(message) + "\n")
        self.output_text.see(tk.END)
        self.output_text.config(state='disabled')

# ==============================================================================
#  3. BOILERPLATE TO RUN THE APP
# ==============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = JiraTrainerApp(root)
    root.mainloop()