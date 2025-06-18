# Step 1: Import necessary libraries
import os          # To interact with the operating system (finding files)
import json        # To read and parse JSON files
import pandas as pd  # The best library for handling data in tables (DataFrames)
from parametars import TEST
# --- Configuration ---
# Set the path to the folder containing your .json files.
# Make sure this path is correct for your system!
# For example: "C:/Users/YourUser/Desktop/project/data"
if not TEST:
    DATA_FOLDER = "data/" 
else:
    DATA_FOLDER = "deploy and test\jsons"

# Step 2: Find all .json files in the data folder
print(f"Searching for .json files in: {DATA_FOLDER}")
try:
    # Use a list comprehension to find all files ending with .json
    json_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".json") and "name_mapping" not in f]
except FileNotFoundError:
    print(f"Error: The directory '{DATA_FOLDER}' was not found.")
    print("Please make sure the DATA_FOLDER variable is set correctly.")
    json_files = [] # Ensure the list is empty so the script exits gracefully

# Loop over each file found
for project_file_name in json_files:
    print(f"\n--- Processing project file: {project_file_name} ---")

    # We will add the logic for each file inside this loop in the next steps...
        # Create the full path to the file
    file_path = os.path.join(DATA_FOLDER, project_file_name)

    # Open and load the JSON data
    with open(file_path, 'r', encoding='utf-8') as f:
        issues_data = json.load(f)

    # --- This is our more efficient approach ---
    # 1. Create an empty list to hold all our processed issue dictionaries
    processed_issues = []

    # Loop through each issue in the loaded data
    for raw_issue in issues_data:
        # We will process each issue here in the next step

    # After the loop, we will create the DataFrame and save it.
            # Create a dictionary to hold the clean data for a single issue
        clean_issue = {}

        # --- Extract and transform the data ---
        
        # 1. Simple fields (just copy them over)
        clean_issue["id"] = raw_issue["_id"]
        clean_issue["summary"] = raw_issue["summary"]
        clean_issue["description"] = raw_issue.get("description", "") # Use .get() for safety
        clean_issue["project_name"] = raw_issue["projectname"]
        clean_issue["created_date"] = raw_issue["created"]
        clean_issue["assignee"] = raw_issue["assignee"]

        # 2. Nested fields (extract the 'name' from the inner dictionary)
        clean_issue["issue_type_name"] = raw_issue["issuetype"]["name"]
        clean_issue["priority_name"] = raw_issue["priority"]["name"]

        # 3. List fields (join them into a single string or extract names)
        
        # Join all labels into a single space-separated string
        clean_issue["labels"] = " ".join(raw_issue["labels"])

        # Extract the 'name' from each dictionary in the components list
        component_names = [comp["name"] for comp in raw_issue["components"]]
        clean_issue["components_name"] = component_names # Store as a list
        
        # --- Add the processed dictionary to our list ---
        processed_issues.append(clean_issue)
    # --- Create the DataFrame and save it ---
    
    # Check if we actually processed any issues
    if not processed_issues:
        print(f"No issues found in {project_file_name}. Skipping.")
        continue # Go to the next project file

    # 1. Create the DataFrame from our list of dictionaries (very fast!)
    df = pd.DataFrame(processed_issues)

    # 2. Factorize the 'assignee' column to convert names to numbers
    # This is the critical step for the classifier's target variable (y)
    # Factorize the 'assignee' column but keep the mapping
    df["assignee_num"], name_mapping = pd.factorize(df["assignee"])

    # Save the name mapping for later use
    name_mapping = pd.Series(name_mapping)
    name_mapping.to_csv(os.path.join(DATA_FOLDER, f"{project_file_name}_name_mapping.csv"), index=False)

    # 3. Determine the output CSV filename
    # We get the project name from the first issue we processed
    project_name = df["project_name"][0]
    if not TEST:
        output_filename = os.path.join(DATA_FOLDER, f"{project_name}.csv")
    else:
        output_filename = os.path.join("deploy and test\csvs", f"{project_name}_test.csv")
    # 4. Save the final DataFrame to a CSV file
    df.to_csv(output_filename, index=False)
    print(f"Successfully processed {len(df)} issues.")
    print(f"Output saved to: {output_filename}")