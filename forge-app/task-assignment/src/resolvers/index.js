import Resolver from '@forge/resolver';
import api, { route } from '@forge/api';

const resolver = new Resolver();

// Your AWS API Gateway endpoint
const AWS_API_ENDPOINT = 'https://0sz41t116d.execute-api.us-east-1.amazonaws.com/dev/assign';

// Call the real AWS Lambda ML model
const callMLPredictionAPI = async (issueData) => {
  console.log('Calling AWS Lambda ML model for:', issueData);
  
  try {
    // Prepare the payload for the Lambda function
    const payload = {
      action: 'predict',
      issue: {
        summary: issueData.summary || '',
        description: issueData.description || '',
        labels: issueData.labels || '',
        components_name: issueData.components || '',
        priority_name: issueData.priority || 'Medium',
        issue_type_name: issueData.issueType || 'Task'
      }
    };

    console.log('Sending payload to Lambda:', payload);

    // Make HTTP request to AWS API Gateway
    const response = await api.fetch(AWS_API_ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      throw new Error(`API call failed with status: ${response.status}`);
    }

    const result = await response.json();
    console.log('AWS Lambda response:', result);

    // Check if the response structure is as expected
    if (result.success && result.prediction && result.prediction.recommendations) {
      // Transform the Lambda response to match our expected format
      const recommendations = result.prediction.recommendations.map((assignee, index) => ({
        assignee: assignee,
        confidence: 0.9 - (index * 0.1), // Decreasing confidence for ranking
        reasoning: `AI Model Recommendation #${index + 1} - Based on ${result.prediction.model_type}`,
        skills: ['AI Predicted'], // We don't have detailed skills from the model
        experience: 'Unknown',
        workload: 'Unknown'
      }));

      return {
        predictions: recommendations,
        modelInfo: {
          modelType: result.prediction.model_type || 'My_Super_Enhanced_Model',
          version: '1.0',
          trainingDate: 'Latest',
          totalFeatures: 'Multiple',
          accuracy: 'Production Model'
        },
        source: 'AWS Lambda ML Model',
        apiResponse: result
      };
    } else {
      throw new Error('Invalid response format from Lambda');
    }

  } catch (error) {
    console.error('Error calling AWS Lambda ML API:', error);
    
    // Fallback to a simple error response
    return {
      predictions: [{
        assignee: 'unassigned',
        confidence: 0.1,
        reasoning: `API Error: ${error.message}`,
        skills: [],
        experience: 'Unknown',
        workload: 'Unknown'
      }],
      modelInfo: {
        modelType: 'Error Fallback',
        version: 'N/A',
        trainingDate: 'N/A',
        totalFeatures: 'N/A',
        accuracy: 'N/A'
      },
      source: 'Error Fallback',
      error: error.message
    };
  }
};

resolver.define('getIssueData', async (req) => {
  try {
    console.log('Getting issue data for context:', req.context);
    
    // Check if we have the issue key in context
    if (!req.context?.extension?.issue?.key) {
      throw new Error('No issue key found in context');
    }
    
    const issueKey = req.context.extension.issue.key;
    console.log('Issue key:', issueKey);
    
    // Get issue details from Jira API using asUser() instead of asApp()
    const response = await api.asUser().requestJira(route`/rest/api/3/issue/${issueKey}`, {
      headers: {
        'Accept': 'application/json'
      }
    });

    if (!response.ok) {
      throw new Error(`Jira API returned ${response.status}: ${response.statusText}`);
    }

    const issue = await response.json();
    
    // Extract relevant data for the ML model with better null checking
    const issueData = {
      key: issue.key,
      summary: issue.fields?.summary || '',
      description: extractDescription(issue.fields?.description) || '',
      issueType: issue.fields?.issuetype?.name || '',
      priority: issue.fields?.priority?.name || 'Medium',
      labels: issue.fields?.labels?.map(label => label.name).join(', ') || '',
      components: issue.fields?.components?.map(comp => comp.name).join(', ') || '',
      assignee: issue.fields?.assignee?.displayName || null,
      created: issue.fields?.created,
      updated: issue.fields?.updated
    };

    console.log('Extracted issue data:', issueData);
    return issueData;
    
  } catch (error) {
    console.error('Error getting issue data:', error);
    // Return a more user-friendly error
    throw new Error(`Failed to load issue data: ${error.message}`);
  }
});

// Helper function to extract text from description
function extractDescription(description) {
  if (!description) return '';
  
  try {
    // Handle Atlassian Document Format (ADF)
    if (description.content && Array.isArray(description.content)) {
      return description.content
        .map(block => {
          if (block.content && Array.isArray(block.content)) {
            return block.content
              .map(item => item.text || '')
              .join(' ');
          }
          return '';
        })
        .join(' ')
        .trim();
    }
    
    // Handle plain text
    if (typeof description === 'string') {
      return description;
    }
    
    return '';
  } catch (err) {
    console.warn('Error extracting description:', err);
    return '';
  }
}

resolver.define('predictTaskAssignment', async (req) => {
  try {
    console.log('Predicting task assignment for:', req.payload.issueData);
    
    const { issueData } = req.payload;
    
    // Prepare data for the AWS Lambda function
    const requestBody = {
      issue: {
        summary: issueData.summary,
        description: issueData.description,
        labels: issueData.labels,
        components_name: issueData.components,
        priority_name: issueData.priority,
        issue_type_name: issueData.issueType
      },
      action: 'predict'
    };

    console.log('Request body prepared:', requestBody);

    // TODO: Replace with actual AWS API call when ready
    // const response = await fetch(AWS_API_ENDPOINT, {
    //   method: 'POST',
    //   headers: {
    //     'Content-Type': 'application/json',
    //   },
    //   body: JSON.stringify(requestBody)
    // });

    // Call the real AWS Lambda ML model
    console.log('Calling AWS Lambda ML model...');
    const result = await callMLPredictionAPI(issueData);
    
    console.log('Prediction result:', result);

    // Format the response for the frontend
    return {
      recommendations: result.predictions || [],
      modelInfo: result.modelInfo || {
        version: 'v1.0.0-simulated',
        lastTrained: new Date().toISOString(),
        accuracy: 0.5
      },
      confidence: result.confidence || 0.5,
      isSimulated: true // Flag to indicate this is simulated data
    };

  } catch (error) {
    console.error('Error predicting task assignment:', error);
    throw new Error('Failed to predict task assignment: ' + error.message);
  }
});

resolver.define('assignTask', async (req) => {
  try {
    console.log('Assigning task:', req.payload);
    
    const { issueKey, assignee, confidence, reasoning } = req.payload;
    
    // Get the assignee's account ID from their display name
    const userSearchResponse = await api.asApp().requestJira(route`/rest/api/3/user/search?query=${assignee}`, {
      headers: {
        'Accept': 'application/json'
      }
    });

    const users = await userSearchResponse.json();
    const user = users.find(u => u.displayName === assignee);
    
    if (!user) {
      throw new Error(`User ${assignee} not found`);
    }

    // Assign the issue
    const assignResponse = await api.asApp().requestJira(route`/rest/api/3/issue/${issueKey}/assignee`, {
      method: 'PUT',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        accountId: user.accountId
      })
    });

    if (!assignResponse.ok) {
      throw new Error(`Failed to assign issue: ${assignResponse.status} ${assignResponse.statusText}`);
    }

    // Add a comment with AI reasoning
    try {
      await api.asApp().requestJira(route`/rest/api/3/issue/${issueKey}/comment`, {
        method: 'POST',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          body: {
            type: 'doc',
            version: 1,
            content: [
              {
                type: 'paragraph',
                content: [
                  {
                    type: 'text',
                    text: `🤖 AI Task Assignment: Assigned to ${assignee} with ${(confidence * 100).toFixed(1)}% confidence. ${reasoning}`
                  }
                ]
              }
            ]
          }
        })
      });
    } catch (commentError) {
      console.warn('Failed to add comment, but assignment succeeded:', commentError);
    }

    // TODO: Replace with actual AWS API call when ready
    // Optional: Update the ML model with this assignment (partial_fit)
    // const updateBody = {
    //   issue: {
    //     summary: req.context.extension.issue.summary,
    //     assignee: assignee
    //   },
    //   action: 'partial_fit'
    // };

    // try {
    //   await fetch(AWS_API_ENDPOINT, {
    //     method: 'POST',
    //     headers: {
    //       'Content-Type': 'application/json',
    //     },
    //     body: JSON.stringify(updateBody)
    //   });
    // } catch (updateError) {
    //   console.warn('Failed to update model, but assignment succeeded:', updateError);
    // }

    console.log('Assignment successful - simulated model update');

    return { success: true, assignee, confidence, reasoning };

  } catch (error) {
    console.error('Error assigning task:', error);
    throw new Error('Failed to assign task: ' + error.message);
  }
});

// Debug resolver to test basic functionality
resolver.define('testConnection', async (req) => {
  try {
    console.log('Testing connection...', req.context);
    
    return {
      success: true,
      message: 'Connection working!',
      context: req.context ? 'Context available' : 'No context',
      timestamp: new Date().toISOString()
    };
  } catch (error) {
    console.error('Test connection error:', error);
    return {
      success: false,
      error: error.message
    };
  }
});

export const handler = resolver.getDefinitions();
