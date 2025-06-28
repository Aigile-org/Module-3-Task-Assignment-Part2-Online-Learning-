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
      console.log('🔄 Converting account IDs to display names...');
      
      // Get account IDs from recommendations
      const accountIds = result.prediction.recommendations;
      console.log('Account IDs received:', accountIds);
      
      // Convert account IDs to display names
      const displayNames = await resolveAccountIdsToNames(accountIds);
      console.log('Display names resolved:', displayNames);
      
      // Transform the Lambda response to match our expected format with display names
      const recommendations = displayNames.map((displayName, index) => ({
        assignee: displayName,
        accountId: accountIds[index], // Keep account ID for assignment
        confidence: 0.9 - (index * 0.1), // Decreasing confidence for ranking
        reasoning: `AI Model Recommendation #${index + 1} - Based on ${result.prediction.model_type}`,
        skills: ['AI Predicted'], // We don't have detailed skills from the model
        experience: 'Unknown',
        workload: 'Unknown'
      }));

      return {
        predictions: recommendations,
        modelInfo: {
          modelType: result.prediction.model_type || 'Enhanced_Adaboost',
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

// Call the partial_fit API to update the ML model with assignment feedback
const callPartialFitAPI = async (issueData, assigneeAccountId) => {
  console.log('🔄 Starting partial_fit API call...');
  console.log('📊 Issue data received:', JSON.stringify(issueData, null, 2));
  console.log('👤 Assignee:', assigneeAccountId);
  
  // Validate input data
  if (!issueData || typeof issueData !== 'object') {
    console.error('❌ Invalid issue data:', issueData);
    return {
      success: false,
      error: 'Invalid issue data provided'
    };
  }
  
  if (!assigneeAccountId) {
    console.error('❌ No assignee provided');
    return {
      success: false,
      error: 'No assignee provided'
    };
  }
  
  try {
    // Prepare the payload for the Lambda function
    const payload = {
      action: 'partial_fit',
      issue: {
        summary: issueData.summary || '',
        description: issueData.description || '',
        labels: issueData.labels || '',
        components_name: issueData.components || '',
        priority_name: issueData.priority || 'Medium',
        issue_type_name: issueData.issueType || 'Task'
      },
      assignee: assigneeAccountId
    };

    console.log('📤 Sending partial_fit payload to Lambda:');
    console.log(JSON.stringify(payload, null, 2));
    console.log('🔍 Payload validation:');
    console.log('  - action:', payload.action);
    console.log('  - assignee:', payload.assignee);
    console.log('  - issue object:', payload.issue);
    console.log('  - issue summary:', payload.issue.summary);
    console.log('  - issue empty check:', !payload.issue);
    console.log('  - issue object keys:', Object.keys(payload.issue));
    console.log('🌐 API Endpoint:', AWS_API_ENDPOINT);

    // Make HTTP request to AWS API Gateway
    const response = await api.fetch(AWS_API_ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload)
    });

    console.log('📥 HTTP Response Status:', response.status);
    console.log('📥 HTTP Response OK:', response.ok);

    if (!response.ok) {
      const errorText = await response.text();
      console.error('❌ Partial fit API HTTP error:', response.status, errorText);
      throw new Error(`Partial fit API call failed with status: ${response.status} - ${errorText}`);
    }

    const result = await response.json();
    console.log('✅ Partial fit API successful response:');
    console.log(JSON.stringify(result, null, 2));
    
    return {
      success: true,
      result: result,
      httpStatus: response.status
    };

  } catch (error) {
    console.error('💥 Error calling partial_fit API:', error);
    console.error('💥 Error stack:', error.stack);
    return {
      success: false,
      error: error.message,
      errorType: error.constructor.name
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

    // Prepare issue data for partial_fit API call
    const issue = req.context.extension.issue;
    const issueDataForLearning = {
      summary: issue.fields?.summary || '',
      description: issue.fields?.description || '',
      issueType: issue.fields?.issuetype?.name || 'Task',
      priority: issue.fields?.priority?.name || 'Medium',
      labels: issue.fields?.labels?.map(label => label.name).join(', ') || '',
      components: issue.fields?.components?.map(comp => comp.name).join(', ') || ''
    };

    // Call partial_fit API to update the model with this assignment (non-blocking)
    // Don't await this call to avoid slowing down the assignment response
    callPartialFitAPI(issueDataForLearning, user.accountId)
      .then(result => {
        console.log('✅ Partial fit API completed successfully:', result);
      })
      .catch(error => {
        console.warn('❌ Partial fit API failed (but assignment was successful):', error);
      });

    console.log('🚀 Assignment completed immediately, partial_fit running in background');

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

// Get all project members for dropdown
resolver.define('getProjectMembers', async (req) => {
  try {
    console.log('Getting project members...');
    
    // Get the project context
    const projectId = req.context.extension.project.id;
    console.log('Project ID:', projectId);
    
    // Get the issue key to check assignable users for this specific issue
    const issueKey = req.context.extension.issue?.key;
    console.log('Issue key for assignable check:', issueKey);
    
    let assignableUsers = [];
    if (issueKey) {
      // Get users who can be assigned to this specific issue
      console.log('Fetching assignable users for this specific issue...');
      const assignableResponse = await api.asUser().requestJira(route`/rest/api/3/user/assignable/search?issueKey=${issueKey}&maxResults=100`);
      
      if (assignableResponse.ok) {
        assignableUsers = await assignableResponse.json();
        console.log('Found assignable users for this issue:', assignableUsers.length);
        
        const members = assignableUsers.map(user => ({
          accountId: user.accountId,
          displayName: user.displayName,
          active: user.active !== false // Assume active if not specified
        })).filter(user => user.active);
        
        console.log('Processed assignable members:', members.length);
        
        return {
          success: true,
          members: members
        };
      } else {
        console.warn('Failed to get assignable users for issue, falling back to project members');
      }
    }
    
    // Fallback: Get all users who can be assigned to issues in this project
    console.log('Fetching general project assignable users...');
    const response = await api.asUser().requestJira(route`/rest/api/3/user/assignable/search?project=${projectId}&maxResults=100`);
    
    if (response.ok) {
      const users = await response.json();
      console.log('Found project members:', users.length);
      
      // Return simplified user data without email addresses
      const members = users.map(user => ({
        accountId: user.accountId,
        displayName: user.displayName,
        active: user.active
      })).filter(user => user.active); // Only active users
      
      console.log('Processed members:', members.length);
      
      return {
        success: true,
        members: members
      };
    } else {
      const errorText = await response.text();
      console.error('Failed to get project members:', response.status, errorText);
      return {
        success: false,
        error: `Failed to fetch project members: ${response.status} - ${errorText}`
      };
    }
  } catch (error) {
    console.error('Error getting project members:', error);
    return {
      success: false,
      error: error.message
    };
  }
});

// Assign issue to selected user
resolver.define('assignIssue', async (req) => {
  try {
    const { issueKey, assigneeAccountId } = req.payload;
    console.log(`Assigning issue ${issueKey} to user ${assigneeAccountId}`);
    console.log('assigneeAccountId type:', typeof assigneeAccountId);
    console.log('assigneeAccountId value:', assigneeAccountId);
    
    // Validate inputs
    if (!issueKey || !assigneeAccountId) {
      return {
        success: false,
        error: 'Missing required parameters: issueKey or assigneeAccountId'
      };
    }
    
    // Ensure assigneeAccountId is a string and not empty
    const accountId = String(assigneeAccountId).trim();
    if (!accountId || accountId === 'undefined' || accountId === 'null') {
      return {
        success: false,
        error: `Invalid account ID: "${assigneeAccountId}" (type: ${typeof assigneeAccountId})`
      };
    }
    
    console.log('Validated accountId:', accountId);
    
    // First, try to get the issue to verify it exists and check permissions
    const issueResponse = await api.asUser().requestJira(route`/rest/api/3/issue/${issueKey}`, {
      headers: {
        'Accept': 'application/json'
      }
    });
    
    if (!issueResponse.ok) {
      const errorText = await issueResponse.text();
      console.error('Cannot access issue:', issueResponse.status, errorText);
      return {
        success: false,
        error: `Cannot access issue ${issueKey}: ${issueResponse.status}`
      };
    }
    
    // Try assignment using the dedicated assignee endpoint
    console.log('Attempting assignment using assignee endpoint...');
    
    const assignResponse = await api.asUser().requestJira(route`/rest/api/3/issue/${issueKey}/assignee`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify({
        accountId: accountId
      })
    });
    
    if (assignResponse.ok) {
      console.log('Assignment successful!');
      
      // Initialize partialFitResult variable in the correct scope
      let partialFitResult = { success: false, error: 'Not attempted' };
      
      // Get the issue data for partial_fit learning
      try {
        const issue = await issueResponse.json();
        const issueData = {
          key: issue.key,
          summary: issue.fields?.summary || '',
          description: extractDescription(issue.fields?.description) || '',
          issueType: issue.fields?.issuetype?.name || '',
          priority: issue.fields?.priority?.name || 'Medium',
          labels: issue.fields?.labels?.map(label => label.name).join(', ') || '',
          components: issue.fields?.components?.map(comp => comp.name).join(', ') || ''
        };
        
        // Call partial_fit API to help the model learn from this assignment (non-blocking)
        console.log('========== CALLING PARTIAL_FIT API ==========');
        console.log('Issue data for learning:', JSON.stringify(issueData, null, 2));
        console.log('Assignee account ID:', accountId);
        console.log('============================================');
        
        // Don't await this call to avoid slowing down the assignment response
        callPartialFitAPI(issueData, accountId)
          .then(result => {
            console.log('========== PARTIAL_FIT RESPONSE ==========');
            console.log('✅ Partial fit result:', JSON.stringify(result, null, 2));
            console.log('✅ Model learning successful via partial_fit API');
            console.log('✅ Lambda response:', JSON.stringify(result.result, null, 2));
            console.log('=========================================');
          })
          .catch(error => {
            console.log('========== PARTIAL_FIT ERROR ==========');
            console.warn('❌ Model learning failed, but assignment was successful');
            console.warn('❌ Error:', error);
            console.log('======================================');
          });
        
        console.log('🚀 Assignment completed immediately, partial_fit running in background');
        partialFitResult = { success: true, message: 'Partial fit initiated in background' };
        
      } catch (learningError) {
        console.warn('Failed to extract issue data for learning, but assignment was successful:', learningError);
        partialFitResult = { success: false, error: `Learning extraction failed: ${learningError.message}` };
      }
      
      return {
        success: true,
        message: `Successfully assigned issue ${issueKey} to user ${accountId}`,
        partialFitResult: partialFitResult
      };
    } else {
      const errorText = await assignResponse.text();
      console.error('Assignment failed:', assignResponse.status, errorText);
      return {
        success: false,
        error: `Assignment failed: ${assignResponse.status}`,
        details: {
          status: assignResponse.status,
          responseBody: errorText
        }
      };
    }
    
  } catch (error) {
    console.error('Error assigning issue:', error);
    return {
      success: false,
      error: error.message
    };
  }
});

// Helper function to get user display name from account ID
const getUserDisplayName = async (accountId) => {
  try {
    console.log(`🔍 Looking up display name for account ID: ${accountId}`);
    
    // Use Jira's user API to get user details by account ID
    const userResponse = await api.asApp().requestJira(route`/rest/api/3/user?accountId=${accountId}`, {
      headers: {
        'Accept': 'application/json'
      }
    });
    
    if (userResponse.ok) {
      const user = await userResponse.json();
      console.log(`✅ Found user: ${user.displayName} (${user.emailAddress})`);
      return user.displayName || user.emailAddress || accountId;
    } else {
      console.warn(`⚠️ Could not find user for account ID: ${accountId}`);
      return accountId; // Fallback to account ID
    }
  } catch (error) {
    console.error(`❌ Error getting user display name for ${accountId}:`, error);
    return accountId; // Fallback to account ID
  }
};

// Helper function to convert multiple account IDs to display names
const resolveAccountIdsToNames = async (accountIds) => {
  const namePromises = accountIds.map(accountId => getUserDisplayName(accountId));
  return await Promise.all(namePromises);
};

export const handler = resolver.getDefinitions();
