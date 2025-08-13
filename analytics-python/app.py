from fastapi import FastAPI

app = FastAPI()

@app.get("/status")
async def get_status():
    """Returns the operational status of the analytics service."""
    print("Request received for analytics status check.")
    response = {
        "status": "Python analytics service is running successfully"
    }
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)