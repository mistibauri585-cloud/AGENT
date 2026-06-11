from fastapi import FastAPI

app = FastAPI(title="Appna Bank AI")

@app.get("/")
def home():
    return {"status": "Running", "message": "Welcome to Appna Bank AI Backend!"}
