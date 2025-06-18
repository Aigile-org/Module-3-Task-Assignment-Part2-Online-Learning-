// In src/frontend/index.jsx
import React, { useState } from 'react';
import ForgeReconciler, { Text, Button, Code } from '@forge/react';
import { invoke } from '@forge/bridge';

const App = () => {
  const [issueData, setIssueData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const fetchAndDisplayIssueData = async () => {
    setIsLoading(true);
    setIssueData(null); 
    try {
      const data = await invoke('getIssueData');
      
      // --- NEW: Check if the returned data is empty ---
      // This handles cases where the backend might return null or an empty object.
      if (!data || Object.keys(data).length === 0) {
        // If no data is returned, set the specific warning message.
        setIssueData('Warning: No data was returned for this issue.');
      } else {
        // If we have data, format and display it as usual.
        setIssueData(JSON.stringify(data, null, 2));
      }

    } catch (error) {
      // This catch block will still handle network errors or errors from the resolver.
      setIssueData('An error occurred. Please check the developer console for details.');
      console.error('Frontend: Caught an error:', error);
    }
    setIsLoading(false);
  };

  return (
    <>
      <Text>Click the button to fetch the data for the current Jira issue.</Text>
      <Button
        text=" Get Issue Info Get Issue Info Get Issue Info "
        onClick={fetchAndDisplayIssueData}
        disabled={isLoading}
      />
      {isLoading && <Text>Loading...</Text>}
      
      {/* This part will now display either the JSON data or your new warning message.
        We use the <Code> component for the JSON and <Text> for the warning.
      */}
     {issueData && <Text>{issueData}</Text>}
    </>
  );
};

ForgeReconciler.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);