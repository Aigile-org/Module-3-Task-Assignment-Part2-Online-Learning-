# AI Task Assignment Forge App

🤖 An intelligent Jira issue action that uses machine learning to recommend the best team members for task assignment.

## Features

- **Smart Recommendations**: AI analyzes issue details (summary, description, type, priority) to suggest optimal assignees
- **Confidence Scoring**: Each recommendation comes with a confidence percentage
- **Skill Matching**: Shows relevant skills and experience for each recommendation
- **Workload Awareness**: Considers current team member availability
- **One-Click Assignment**: Assign tasks directly from the recommendation interface
- **AI Comments**: Automatically adds reasoning comments when assignments are made

## Current Status

### 🧪 **Simulation Mode (Current)**
- Uses simulated ML predictions for demo purposes
- Realistic mock data with intelligent scoring based on issue content
- Perfect for testing and development

### 🚀 **Production Mode (Ready for AWS API)**
- Prepared for AWS API Gateway integration
- Lambda function template included
- Easy switch to real ML model

## How It Works

1. **Issue Analysis**: Extracts issue data (summary, description, type, priority, labels, components)
2. **ML Prediction**: Sends data to ML model (currently simulated)
3. **Smart Scoring**: Analyzes team members based on:
   - Skills matching with issue requirements
   - Experience level
   - Current workload
   - Historical assignment success
4. **Recommendations**: Returns top 3 candidates with confidence scores
5. **Assignment**: One-click assignment with AI reasoning comment

## Usage

1. Navigate to any Jira issue
2. Click the "task-assignment" action in the issue actions menu
3. Click "Get AI Recommendations" 
4. Review the recommended assignees with their confidence scores and reasoning
5. Click "Assign" on your preferred recommendation
6. The issue is assigned and an AI comment is added explaining the reasoning

## Quick Start

- Build and deploy your app by running:
```
forge deploy
```

- Install your app in an Atlassian site by running:
```
forge install
```

- Develop your app by running `forge tunnel` to proxy invocations locally:
```
forge tunnel
```

## Switching to Production API

### Step 1: Update the AWS API Endpoint
In `src/resolvers/index.js`, update the AWS_API_ENDPOINT:

```javascript
const AWS_API_ENDPOINT = 'https://your-actual-api-gateway-url.amazonaws.com/prod';
```

### Step 2: Enable Real API Calls
In the `predictTaskAssignment` function, uncomment the AWS API call and comment out the simulation:

```javascript
// Comment out simulation
// const result = await simulateMLPrediction(issueData);

// Uncomment real API call
const response = await fetch(AWS_API_ENDPOINT, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify(requestBody)
});
```

## AWS Lambda Expected Response Format

Your AWS Lambda function should return this format:

```json
{
  "predictions": [
    {
      "assignee": "Team Member Name",
      "confidence": 0.92,
      "reasoning": "Strong match for JavaScript and React requirements",
      "skills": ["JavaScript", "React", "Frontend"],
      "experience": 5,
      "workload": 0.6
    }
  ],
  "modelInfo": {
    "version": "v1.2.3",
    "lastTrained": "2024-01-15T10:30:00Z",
    "accuracy": 0.87
  },
  "confidence": 0.92
}
```

## File Structure
```
src/
├── frontend/
│   └── index.jsx          # React UI component
├── resolvers/
│   └── index.js           # Backend API handlers (with simulation)
└── manifest.yml           # Forge app configuration
```

## Support

See [Get help](https://developer.atlassian.com/platform/forge/get-help/) for how to get help and provide feedback.
