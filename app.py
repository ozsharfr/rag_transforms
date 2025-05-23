from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
import uvicorn

from main import main  # Import your existing main function

app = FastAPI()

@app.get("/run")
async def run_rag():
    try:
        # Run your main processing function
        #await run_in_threadpool(main) # Run the main function in a thread pool
        main()
        return JSONResponse(content={"status": "success", "message": "RAG process completed."})
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)