from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from main import main
import io
import logging
from contextlib import redirect_stdout
import os
import uvicorn

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def homepage():
    return """
    <html>
    <head>
        <title>RAG PubMed Interface</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                max-width: 1000px; 
                margin: 0 auto; 
                padding: 20px; 
            }
            .query-input { 
                width: 400px; 
                padding: 10px; 
                margin-right: 10px; 
            }
            .run-button { 
                padding: 10px 20px; 
                background: #4caf50; 
                color: white; 
                border: none; 
                cursor: pointer; 
                border-radius: 4px;
            }
            .run-button:hover { 
                background: #45a049; 
            }
            .run-button:disabled {
                background: #cccccc;
                cursor: not-allowed;
            }
            .conversation-item {
                margin: 20px 0;
                padding: 15px;
                border: 1px solid #ddd;
                border-radius: 8px;
                background: #fafafa;
            }
            .question {
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 10px;
            }
            .answer {
                background: #e8f5e8;
                padding: 15px;
                border-left: 4px solid #4caf50;
                margin: 10px 0;
                white-space: pre-wrap;
            }
            .logs {
                background: #f0f0f0;
                padding: 10px;
                max-height: 200px;
                overflow-y: auto;
                font-size: 0.9em;
                margin-top: 10px;
                white-space: pre-wrap;
            }
            .logs-toggle {
                background: none;
                border: none;
                color: #666;
                cursor: pointer;
                text-decoration: underline;
                font-size: 0.9em;
            }
            .loading {
                color: #ff9800;
                font-style: italic;
            }
            #conversation-history {
                margin-bottom: 30px;
            }
        </style>
    </head>
    <body>
        <h2>PubMed Abstract RAG Pipeline</h2>
        
        <div id="conversation-history"></div>
        
        <div style="position: sticky; bottom: 0; background: white; padding: 20px 0; border-top: 2px solid #eee;">
            <p>Enter your query below:</p>
            <input type="text" id="query" placeholder="e.g., Parkinson treatments" class="query-input">
            <button id="runButton" onclick="runPipeline()" class="run-button">Run RAG</button>
            <button onclick="clearHistory()" style="margin-left: 10px; padding: 10px 15px; background: #ff6b6b; color: white; border: none; border-radius: 4px; cursor: pointer;">Clear History</button>
        </div>

        <script>
            let conversationCount = 0;

            // Allow Enter key to submit
            document.getElementById('query').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    runPipeline();
                }
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
                    <div class="question">Q${conversationCount}: ${query}</div>
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
                        answerDiv.className = 'answer';
                        answerDiv.style.background = '#ffebee';
                        answerDiv.style.borderLeft = '4px solid #f44336';
                        answerDiv.textContent = "Error: " + result.message;
                        
                        // Show error logs
                        logsToggle.style.display = 'inline';
                        logsDiv.textContent = result.message;
                    }
                } catch (error) {
                    const answerDiv = conversationDiv.querySelector('.answer');
                    answerDiv.className = 'answer';
                    answerDiv.style.background = '#ffebee';
                    answerDiv.style.borderLeft = '4px solid #f44336';
                    answerDiv.textContent = "Network Error: " + error;
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
        </script>
    </body>
    </html>
    """

class StringIOHandler(logging.Handler):
    """Custom logging handler that writes to a StringIO object"""
    def __init__(self, string_io):
        super().__init__()
        self.string_io = string_io
        
    def emit(self, record):
        msg = self.format(record)
        self.string_io.write(msg + '\n')

cache = {
    "embeddings": None,
    "chunks": None,
    "document_processed": False
}

@app.get("/run", response_class=JSONResponse)
def run_rag(query: str = "What are Parkinson's treatments?"):
    log_capture = io.StringIO()
    
    try:
        # Get the logger from your main module
        logger = logging.getLogger()  # or use the specific logger name from your config
        
        # Create custom handler to capture logs
        handler = StringIOHandler(log_capture)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        # Add handler temporarily
        logger.addHandler(handler)
        
        # Capture both stdout and the final answer
        stdout_capture = io.StringIO()
        final_answer = None
        
        try:
            with redirect_stdout(stdout_capture):
                if cache["document_processed"]:
                    logger.info("Using cached embeddings - faster processing!")
                    # Call main with cached embeddings
                    embedded_docs , final_answer = main(query=query, log_stream=log_capture, cached_embeddings=cache["embeddings"])
                else:
                    logger.info("First run - processing documents and creating embeddings")
                    # First run - process everything
                    embedded_docs, final_answer = main(query=query, log_stream=log_capture)
                    # Cache the results
                    cache["embeddings"] = embedded_docs
                    cache["document_processed"] = True
                    logger.info("Embeddings cached for future queries")

        except Exception as main_error:
            logger.error(f"Error in main: {str(main_error)}")
            
        # Remove the handler
        logger.removeHandler(handler)
        
        # Get captured content
        logs = log_capture.getvalue()
        stdout_content = stdout_capture.getvalue()
        
        # Extract final answer from stdout if not returned directly
        if not final_answer and stdout_content:
            lines = stdout_content.strip().split('\n')
            for line in lines:
                if line.startswith("Final answer:"):
                    final_answer = line.replace("Final answer:", "").strip()
                    break
        
        return {
            "status": "success", 
            "logs": logs,
            "final_answer": final_answer,
            "stdout": stdout_content
        }
        
    except Exception as e:
        return {
            "status": "error", 
            "message": str(e),
            "logs": log_capture.getvalue()
        }
    finally:
        log_capture.close()

# Launch server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)