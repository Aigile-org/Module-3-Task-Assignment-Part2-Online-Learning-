# Issue Assignment Recommendation System
An online machine learning system that recommends developers for new issues based on historical assignment patterns.


## Setup Instructions

1. Create data folder

2. Prepare training data:
- Place your JSON issue files in the `data/` folder
- Run data processing:
  ```
  python processing_data.py
  ```
This generates CSV training files for each project

## Training Models (Experiment Mode)

1. Configure `running_models.py`:
- Set `MODE = 'e'` (Experiment mode)
- Add models to evaluate in `models_to_test` list
  Example: `models_to_test = [MySuperEnhancedModel, HoeffdingAdaptiveTreeModel]`
- Specify projects in `projects_to_process` (use `.csv` for all projects)

2. Run training: python running_models.py

Trained models will be saved to `saved_models/` folder

## Making Recommendations (Deployment Mode)

1. Prepare test data:
- Create `DEPLOYMENT_TEST.json` in `data/` folder
- Add new issues in JSON format (matching training schema)

2. Process test data: python processing_data.py
This generates `DEPLOYMENT_TEST.csv`

3. Configure `running_models.py`:
- Set `MODE = 'd'` (Deployment mode)
- Set `PROJECT_FOR_TRAINING` to the project name you want to use

4. Get recommendations: python running_models.py
The system will:
- Show 3 recommended developers
- Prompt for the actual assignee
- Update the model with your feedback