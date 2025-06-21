import React, { useEffect, useState } from 'react';
import ForgeReconciler, { Text, Button, SectionMessage, Spinner, Stack, Box, Select } from '@forge/react';
import { invoke } from '@forge/bridge';

const App = () => {
  const [issueData, setIssueData] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [assignmentComplete, setAssignmentComplete] = useState(false);
  const [projectMembers, setProjectMembers] = useState([]);
  const [selectedAssignee, setSelectedAssignee] = useState('');
  const [isAssigning, setIsAssigning] = useState(false);

  // Load issue data when component mounts
  useEffect(() => {
    console.log('Component mounted, loading issue data...');
    loadIssueData();
    loadProjectMembers();
  }, []);

  const loadProjectMembers = async () => {
    try {
      console.log('Loading project members...');
      const response = await invoke('getProjectMembers');
      console.log('getProjectMembers response:', response);
      if (response.success) {
        console.log('Project members loaded successfully:', response.members);
        response.members.forEach((member, index) => {
          console.log(`Member ${index + 1}:`, {
            displayName: member.displayName,
            accountId: member.accountId,
            accountIdType: typeof member.accountId
          });
        });
        setProjectMembers(response.members);
        console.log('Project members set in state, count:', response.members.length);
      } else {
        console.error('Failed to load project members:', response.error);
      }
    } catch (err) {
      console.error('Error loading project members:', err);
    }
  };

  const loadIssueData = async () => {
    try {
      console.log('Starting to load issue data...');
      setLoading(true);
      setError(null);
      
      const data = await invoke('getIssueData');
      console.log('Issue data loaded successfully:', data);
      setIssueData(data);
    } catch (err) {
      console.error('Error loading issue data:', err);
      setError('Failed to load issue data');
    } finally {
      setLoading(false);
    }
  };

  const predictAssignment = async () => {
    if (!issueData) return;
    
    try {
      setLoading(true);
      setError(null);
      
      const result = await invoke('predictTaskAssignment', { issueData });
      setPredictions(result);
    } catch (err) {
      console.error('Error predicting assignment:', err);
      setError('Failed to predict assignment');
    } finally {
      setLoading(false);
    }
  };

  const assignTaskToSelectedUser = async () => {
    if (!selectedAssignee || !issueData) {
      setError('Please select an assignee');
      return;
    }

    try {
      setIsAssigning(true);
      setError(null);
      
      // Debug: Log the selectedAssignee value and type
      console.log('Selected assignee value:', selectedAssignee);
      console.log('Selected assignee type:', typeof selectedAssignee);
      console.log('Is selectedAssignee a string?', typeof selectedAssignee === 'string');
      
      // Handle both string and object cases
      let accountId;
      if (typeof selectedAssignee === 'string') {
        accountId = selectedAssignee;
      } else if (typeof selectedAssignee === 'object' && selectedAssignee.value) {
        accountId = selectedAssignee.value;
      } else if (typeof selectedAssignee === 'object' && selectedAssignee.accountId) {
        accountId = selectedAssignee.accountId;
      } else {
        accountId = String(selectedAssignee);
      }
      
      console.log('Extracted accountId:', accountId);
      console.log('Extracted accountId type:', typeof accountId);
      
      // Ensure accountId is a valid string
      if (!accountId || accountId === 'undefined' || accountId === 'null' || accountId === '[object Object]') {
        setError(`Invalid account ID extracted: "${accountId}". Please try selecting a different user.`);
        return;
      }
      
      console.log('Assigning task with payload:', { 
        issueKey: issueData.key, 
        assigneeAccountId: accountId
      });
      
      const result = await invoke('assignIssue', { 
        issueKey: issueData.key, 
        assigneeAccountId: accountId
      });
      
      console.log('Assignment result:', result);
      
      if (result.success) {
        setAssignmentComplete(true);
        setError(null);
      } else {
        const errorMessage = result.error || 'Failed to assign issue';
        const detailsMessage = result.details ? 
          ` (Status: ${result.details.status}, Details: ${result.details.responseBody})` : '';
        setError(`${errorMessage}${detailsMessage}`);
        console.error('Assignment failed:', result);
      }
    } catch (err) {
      console.error('Error assigning task:', err);
      setError(`Failed to assign task: ${err.message}`);
    } finally {
      setIsAssigning(false);
    }
  };

  if (loading && !issueData) {
    return (
      <Box padding="space.300">
        <Stack space="space.200" alignItems="center">
          <Spinner size="large" />
          <Text size="medium" weight="bold">Loading issue data...</Text>
          <Text color="color.text.subtle">Please wait while we fetch the issue details</Text>
        </Stack>
      </Box>
    );
  }

  if (error) {
    return (
      <Box padding="space.400">
        <SectionMessage appearance="error">
          <Stack space="space.200">
            <Text size="medium" weight="bold">❌ An error occurred</Text>
            <Text color="color.text.subtle">{error}</Text>
            <Button 
              onClick={() => {
                setError(null);
                loadIssueData();
              }}
              appearance="primary"
            >
              🔄 Try Again
            </Button>
          </Stack>
        </SectionMessage>
      </Box>
    );
  }

  if (assignmentComplete) {
    return (
      <Box padding="space.400">
        <SectionMessage appearance="success">
          <Stack space="space.100" alignItems="center">
            <Text size="large">🎉 Task assigned successfully!</Text>
            <Text color="color.text.subtle">The issue has been assigned to the recommended team member.</Text>
          </Stack>
        </SectionMessage>
      </Box>
    );
  }

  return (
    <Box padding="space.400">
      <Stack space="space.200">
        {/* Header Section */}
        <Box >
          <Text size="large" weight="bold" color="color.text.accent.blue">
            🤖 AI Task Assignment
          </Text>
          <Box>
            <Text color="color.text.subtle">
              Get intelligent recommendations for who should work on this issue
            </Text>
          </Box>
        </Box>
        
        {/* Issue Details Section */}
        {issueData && (
          <Box 
            marginTop="space.400"
            padding="space.400" 
            backgroundColor="color.background.accent.blue.subtlest"
            style={{ borderRadius: '8px', border: '1px solid #E3F2FD' }}
          >
            <Stack space="space.50">
              <Text size="medium" weight="bold" color="color.text.accent.blue">
                📋 Issue Details
              </Text>
              <Stack space="space.200">
                <Box marginTop="space.400">
                  <Text weight="medium">🔑 Issue Key</Text>
                  <Box marginTop="space.400">
                    <Text color="color.text.subtle">{issueData.key || 'Unknown'}</Text>
                  </Box>
                </Box>
                <Box>
                  <Text weight="medium">📝 Summary</Text>
                  <Box marginTop="space.050">
                    <Text color="color.text.subtle">{issueData.summary || 'No summary'}</Text>
                  </Box>
                </Box>
                <Box>
                  <Text weight="medium">🏷️ Type</Text>
                  <Box marginTop="space.050">
                    <Text color="color.text.subtle">{issueData.issueType || 'Unknown'}</Text>
                  </Box>
                </Box>
                <Box>
                  <Text weight="medium">⚡ Priority</Text>
                  <Box marginTop="space.050">
                    <Text color="color.text.subtle">{issueData.priority || 'Medium'}</Text>
                  </Box>
                </Box>
              </Stack>
            </Stack>
          </Box>
        )}

        {/* Get Recommendations Button */}
        {!predictions && (
          <Box>
            <Button 
              appearance="primary" 
              onClick={predictAssignment}
              isDisabled={loading || !issueData}
              size="large"
            >
              {loading ? '🔍 Analyzing...' : '🎯 Get AI Recommendations'}
            </Button>
          </Box>
        )}

        {/* Loading State */}
        {loading && predictions === null && (
          <Box>
            <Stack space="space.200" alignItems="center">
              <Spinner size="large" />
              <Box>
                <Text size="medium" weight="bold">🤖 AI is analyzing the issue...</Text>
              </Box>
              <Box>
                <Text color="color.text.subtle">This may take a few seconds</Text>
              </Box>
            </Stack>
          </Box>
        )}

        {/* Predictions Section */}
        {predictions && predictions.recommendations && predictions.recommendations.length > 0 && (
          <Stack space="space.200">
            <Box>
              <Text size="medium" weight="bold" color="color.text.accent.green">
                🎯 AI Recommended Assignees
              </Text>
              <Box marginTop="space.100">
                <Text color="color.text.subtle">
                  Based on issue analysis and team expertise
                </Text>
              </Box>
            </Box>
            
            
            <Stack space="space.100">
              {predictions.recommendations.slice(0, 3).map((rec, index) => {
                const assignee = rec?.assignee || 'Unknown';
                
                return (
                  <Box 
                    key={index} 
                    padding="space.400" 
                    backgroundColor={index === 0 ? "color.background.accent.green.subtlest" : "color.background.neutral"}
                    style={{ 
                      borderRadius: '6px', 
                      border: index === 0 ? '2px solid #E8F5E8' : '1px solid #E5E5E5' 
                    }}
                  >
                    <Text weight="bold" size="medium">
                      {index === 0 ? '🥇' : index === 1 ? '🥈' : '🥉'} #{index + 1} {assignee}
                    </Text>
                  </Box>
                );
              })}
            </Stack>
            
            {/* Assignment Section */}
            <Box 
              padding="space.400" 
              backgroundColor="color.background.accent.blue.subtlest"
              style={{ borderRadius: '8px', border: '1px solid #E3F2FD' }}
            >
              <Stack space="space.100">
                <Box>
                  <Text size="medium" weight="bold" color="color.text.accent.blue">
                    👥 Assign to Team Member
                  </Text>
                  <Box marginTop="space.100">
                    <Text color="color.text.subtle">
                      Choose any team member from your project
                    </Text>
                  </Box>
                </Box>
                
                <Box>
                  <Text weight="medium">Select Assignee:</Text>
                  <Box marginTop="space.100">
                    <Select
                      value={selectedAssignee}
                      onChange={(value) => {
                        console.log('=== SELECT DROPDOWN CHANGED ===');
                        console.log('Dropdown selection changed:', value);
                        console.log('Selection type:', typeof value);
                        console.log('Selection stringified:', JSON.stringify(value));
                        if (typeof value === 'object') {
                          console.log('Object keys:', Object.keys(value));
                          console.log('Object.value:', value.value);
                          console.log('Object.accountId:', value.accountId);
                        }
                        console.log('================================');
                        setSelectedAssignee(value);
                      }}
                      placeholder="-- Select a team member --"
                      options={projectMembers.map((member) => {
                        console.log('Creating option for member:', member);
                        return {
                          label: member.displayName,
                          value: member.accountId
                        };
                      })}
                    />
                  </Box>
                </Box>
                
                <Box>
                  <Button 
                    onClick={assignTaskToSelectedUser}
                    isDisabled={!selectedAssignee || isAssigning}
                    appearance="primary"
                    size="large"
                  >
                    {isAssigning ? '⏳ Assigning...' : '✅ Assign Issue'}
                  </Button>
                </Box>
              </Stack>
            </Box>
          </Stack>
        )}

        {/* No Recommendations */}
        {predictions && predictions.recommendations && predictions.recommendations.length === 0 && (
          <Box>
            <SectionMessage appearance="warning">
              <Stack space="space.50">
                <Text weight="bold">⚠️ No recommendations available</Text>
                <Text>The AI couldn't find suitable assignees for this issue.</Text>
                <Box marginTop="space.200">
                  <Button onClick={predictAssignment} appearance="primary">
                    🔄 Try Again
                  </Button>
                </Box>
              </Stack>
            </SectionMessage>
          </Box>
        )}
      </Stack>
    </Box>
  );
};

ForgeReconciler.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);