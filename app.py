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
    <head><title>RAG PubMed Interface</title></head>
    <body>
        <h2>PubMed Abstract RAG Pipeline</h2>
        <p>Enter your query below:</p>
        <input type="text" id="query" placeholder="e.g., Parkinson treatments" style="width: 400px;">
        <button onclick="runPipeline()">Run RAG</button>
        
        <div style="margin-top: 20px;">
            <h3>Final Answer:</h3>
            <pre id="answer" style="background: #e8f5e8; padding: 1em; border-left: 4px solid #4caf50;"></pre>
            
            <h3>Processing Logs:</h3>
            <pre id="logs" style="background: #f0f0f0; padding: 1em; max-height: 400px; overflow-y: auto;"></pre>
        </div>

        <script>
            async function runPipeline() {
                const query = document.getElementById('query').value;
                document.getElementById('answer').innerText = "Processing...";
                document.getElementById('logs').innerText = "Starting RAG pipeline...";
                
                try {
                    const response = await fetch('/run?query=' + encodeURIComponent(query));
                    const result = await response.json();
                    
                    if (result.status === "success") {
                        document.getElementById('answer').innerText = result.final_answer || "No answer generated";
                        document.getElementById('logs').innerText = result.logs || "No logs available";
                    } else {
                        document.getElementById('answer').innerText = "Error occurred";
                        document.getElementById('logs').innerText = result.message;
                    }
                } catch (error) {
                    document.getElementById('answer').innerText = "Error occurred";
                    document.getElementById('logs').innerText = "Error: " + error;
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
                result = main(query=query, log_stream=log_capture)
                # If main returns the final answer, capture it
                embedded_docs = result
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