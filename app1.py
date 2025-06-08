from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from main import main
import io
import logging
from contextlib import redirect_stdout
import os
import uvicorn

app = FastAPI()

# Mount static files (CSS, JS, images, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Global cache for embeddings
cache = {
    "embeddings": None,
    "chunks": None,
    "document_processed": False
}

class StringIOHandler(logging.Handler):
    """Custom logging handler that writes to a StringIO object"""
    def __init__(self, string_io):
        super().__init__()
        self.string_io = string_io
        
    def emit(self, record):
        msg = self.format(record)
        self.string_io.write(msg + '\n')


@app.get("/", response_class=HTMLResponse)
def homepage(request: Request):
    """Serve the main homepage using Jinja2 template"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": "RAG PubMed Interface",
        "heading": "PubMed Abstract RAG Pipeline",
        "placeholder": "e.g., Parkinson treatments"
    })


@app.get("/run", response_class=JSONResponse)
def run_rag(query: str = "What are Parkinson's treatments?"):
    """Run the RAG pipeline and return results"""
    log_capture = io.StringIO()
    
    try:
        # Get the logger from your main module
        logger = logging.getLogger()
        
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
                    embedded_docs, final_answer = main(
                        query=query, 
                        log_stream=log_capture, 
                        cached_embeddings=cache["embeddings"]
                    )
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