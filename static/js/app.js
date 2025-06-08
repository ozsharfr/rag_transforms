// static/js/app.js

let conversationCount = 0;

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Allow Enter key to submit
    document.getElementById('query').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            runPipeline();
        }
    });
    
    // Focus on input field when page loads
    document.getElementById('query').focus();
});

async function runPipeline() {
    const query = document.getElementById('query').value.trim();
    if (!query) {
        alert('Please enter a query');
        return;
    }

    const runButton = document.getElementById('runButton');
    const queryInput = document.getElementById('query');
    
    // Disable input and button during processing
    runButton.disabled = true;
    runButton.textContent = 'Processing...';
    queryInput.disabled = true;

    conversationCount++;
    const conversationId = 'conversation-' + conversationCount;

    // Create new conversation item
    const conversationDiv = document.createElement('div');
    conversationDiv.className = 'conversation-item';
    conversationDiv.id = conversationId;
    conversationDiv.innerHTML = `
        <div class="question">Q${conversationCount}: ${escapeHtml(query)}</div>
        <div class="answer loading">Processing your query...</div>
        <button class="logs-toggle" onclick="toggleLogs('${conversationId}')" style="display: none;">Show Processing Details</button>
        <div class="logs" id="${conversationId}-logs" style="display: none;"></div>
    `;

    // Add to conversation history
    document.getElementById('conversation-history').appendChild(conversationDiv);

    // Scroll to the new question
    conversationDiv.scrollIntoView({ behavior: 'smooth' });

    // Clear the input for next question
    queryInput.value = '';

    try {
        const response = await fetch('/run?query=' + encodeURIComponent(query));
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        const answerDiv = conversationDiv.querySelector('.answer');
        const logsToggle = conversationDiv.querySelector('.logs-toggle');
        const logsDiv = conversationDiv.querySelector('.logs');
        
        if (result.status === "success") {
            answerDiv.className = 'answer';
            answerDiv.textContent = result.final_answer || "No answer generated";
            
            // Show logs toggle and populate logs
            logsToggle.style.display = 'inline';
            logsDiv.textContent = result.logs || "No logs available";
        } else {
            answerDiv.className = 'answer error';
            answerDiv.textContent = "Error: " + result.message;
            
            // Show error logs
            logsToggle.style.display = 'inline';
            logsDiv.textContent = result.message;
        }
    } catch (error) {
        const answerDiv = conversationDiv.querySelector('.answer');
        answerDiv.className = 'answer error';
        answerDiv.textContent = "Network Error: " + error.message;
        
        console.error('Error:', error);
    } finally {
        // Re-enable input and button
        runButton.disabled = false;
        runButton.textContent = 'Run RAG';
        queryInput.disabled = false;
        queryInput.focus();
    }
}

function toggleLogs(conversationId) {
    const logsDiv = document.getElementById(conversationId + '-logs');
    const toggleButton = document.querySelector(`#${conversationId} .logs-toggle`);
    
    if (logsDiv.style.display === 'none') {
        logsDiv.style.display = 'block';
        toggleButton.textContent = 'Hide Processing Details';
    } else {
        logsDiv.style.display = 'none';
        toggleButton.textContent = 'Show Processing Details';
    }
}

function clearHistory() {
    if (confirm('Are you sure you want to clear the conversation history?')) {
        document.getElementById('conversation-history').innerHTML = '';
        conversationCount = 0;
    }
}

// Utility function to escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}