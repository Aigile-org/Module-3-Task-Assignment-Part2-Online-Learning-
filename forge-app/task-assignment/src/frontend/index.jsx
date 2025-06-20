import React, { useEffect, useState } from 'react';
import ForgeReconciler, { Text, Button, SectionMessage, Spinner, Stack, Box } from '@forge/react';
import { invoke } from '@forge/bridge';

const App = () => {
  const [issueData, setIssueData] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [assignmentComplete, setAssignmentComplete] = useState(false);

  // Load issue data when component mounts
  useEffect(() => {
    console.log('Component mounted, loading issue data...');
    loadIssueData();
  }, []);

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

  const assignTask = async (assignee, confidence, reasoning) => {
    try {
      setLoading(true);
      await invoke('assignTask', { 
        issueKey: issueData?.key || '', 
        assignee: assignee || '',
        confidence: confidence || 0,
        reasoning: reasoning || ''
      });
      setAssignmentComplete(true);
    } catch (err) {
      console.error('Error assigning task:', err);
      setError('Failed to assign task');
    } finally {
      setLoading(false);
    }
  };

  const testConnection = async () => {
    try {
      const testResult = await invoke('testConnection');
      console.log('Test result:', testResult);
      alert('Test connection successful');
    } catch (err) {
      console.error('Test failed:', err);
      alert('Test connection failed');
    }
  };

  if (loading && !issueData) {
    return (
      <Box padding="medium">
        <Stack space="medium">
          <Spinner size="medium" />
          <Text>Loading issue data...</Text>
          <Button appearance="subtle" onClick={testConnection}>
            Test Connection
          </Button>
        </Stack>
      </Box>
    );
  }

  if (error) {
    return (
      <Box padding="medium">
        <SectionMessage appearance="error">
          <Stack space="small">
            <Text>An error occurred</Text>
            <Button onClick={() => {
              setError(null);
              loadIssueData();
            }}>
              Try Again
            </Button>
          </Stack>
        </SectionMessage>
      </Box>
    );
  }

  if (assignmentComplete) {
    return (
      <Box padding="medium">
        <SectionMessage appearance="success">
          <Text>Task assigned successfully!</Text>
        </SectionMessage>
      </Box>
    );
  }

  return (
    <Box padding="medium">
      <Stack space="medium">
        <Text>AI Task Assignment</Text>
        
        {issueData && (
          <Box>
            <Stack space="small">
              <Text>Issue: {issueData.key || 'Unknown'}</Text>
              <Text>Summary: {issueData.summary || 'No summary'}</Text>
              <Text>Type: {issueData.issueType || 'Unknown'}</Text>
              <Text>Priority: {issueData.priority || 'Medium'}</Text>
            </Stack>
          </Box>
        )}

        {!predictions && (
          <Button 
            appearance="primary" 
            onClick={predictAssignment}
            isDisabled={loading || !issueData}
          >
            {loading ? 'Analyzing...' : 'Get AI Recommendations'}
          </Button>
        )}

        {loading && predictions === null && (
          <Box>
            <Spinner size="medium" />
            <Text>AI is analyzing the issue...</Text>
          </Box>
        )}

        {predictions && predictions.recommendations && predictions.recommendations.length > 0 && (
          <Box>
            <Text>AI Recommended Assignees:</Text>
            {predictions.isSimulated && (
              <SectionMessage appearance="information">
                <Text>Using simulated ML model for demo</Text>
              </SectionMessage>
            )}
            <Stack space="small">
              {predictions.recommendations.map((rec, index) => {
                const assignee = rec?.assignee || 'Unknown';
                const confidence = rec?.confidence || 0;
                const reasoning = rec?.reasoning || 'No reasoning';
                
                return (
                  <Box key={index} padding="small" backgroundColor="neutral">
                    <Stack space="small">
                      <Text>#{index + 1} {assignee}</Text>
                      <Text>Confidence: {Math.round(confidence * 100)}%</Text>
                      <Text>Reason: {reasoning}</Text>
                      <Button 
                        onClick={() => assignTask(assignee, confidence, reasoning)}
                        isDisabled={loading}
                        appearance={index === 0 ? "primary" : "default"}
                      >
                        Assign
                      </Button>
                    </Stack>
                  </Box>
                );
              })}
            </Stack>
            
            <Box marginTop="medium">
              <Button 
                onClick={() => {
                  setPredictions(null);
                  setError(null);
                }}
                appearance="subtle"
              >
                Get New Recommendations
              </Button>
            </Box>
          </Box>
        )}

        {predictions && predictions.recommendations && predictions.recommendations.length === 0 && (
          <Box>
            <SectionMessage appearance="warning">
              <Text>No recommendations available</Text>
              <Button onClick={predictAssignment}>Try Again</Button>
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
