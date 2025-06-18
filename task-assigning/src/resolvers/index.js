import Resolver from '@forge/resolver';
import api, { asApp, route } from '@forge/api';

// --- HELPER FUNCTION ---
// This function safely navigates the complex description object to extract plain text.
const extractTextFromDescription = (descriptionObject) => {
    if (!descriptionObject || !descriptionObject.content) {
        return ""; // Return empty string if no description
    }
    
    let textContent = [];
    // Loop through each content block
    descriptionObject.content.forEach(block => {
        if (block.content) {
            // Loop through each item in the block
            block.content.forEach(item => {
                if (item.type === 'text' && item.text) {
                    textContent.push(item.text);
                }
            });
        }
    });
    // Join all text parts with a newline
    return textContent.join('\n');
};


const resolver = new Resolver();

resolver.define('getIssueData', async (req) => {
  try {
    const issueId = req.context.extension.issue.id;
    if (!issueId) {
      throw new Error("Could not find the issue ID in the context.");
    }
    
    const response = await api.asApp().requestJira(route`/rest/api/3/issue/${issueId}`);
    if (!response.ok) {
        throw new Error(`Failed to fetch issue data. Status: ${response.status}`);
    }

    // This is the full, raw data from the Jira API
    const rawIssueData = await response.json();

    // --- TRANSFORMATION LOGIC STARTS HERE ---
    // Create a new, clean object based on your desired format.
    const transformedIssue = {
        _id: rawIssueData.id,
        key: rawIssueData.key,
        // Safely get the assignee's name, or null if unassigned
        assignee: rawIssueData.fields.assignee ? rawIssueData.fields.assignee.displayName : null,
        components: rawIssueData.fields.components,
        // Format the date to be more readable
        created: new Date(rawIssueData.fields.created).toLocaleString('sv-SE'), // 'sv-SE' gives YYYY-MM-DD HH:MM:SS format
        // Use our helper function for the description
        description: extractTextFromDescription(rawIssueData.fields.description),
        issuetype: rawIssueData.fields.issuetype,
        labels: rawIssueData.fields.labels,
        priority: rawIssueData.fields.priority,
        // Safely get the project name
        projectname: rawIssueData.fields.project ? rawIssueData.fields.project.name : null,
        summary: rawIssueData.fields.summary
    };
    
    // --- RETURN THE TRANSFORMED DATA ---
    // The frontend will now receive this clean object instead of the raw one.
    return transformedIssue;

  } catch (error) {
    console.error("An error occurred while fetching issue data:", error);
    throw error;
  }
});

export const handler = resolver.getDefinitions();