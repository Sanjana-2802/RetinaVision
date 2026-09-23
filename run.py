import uvicorn

if __name__ == "__main__":
    print("==================================================================")
    print("AI-BASED EYE DISEASE IDENTIFICATION & HEALTH RECOMMENDATION SYSTEM")
    print("BATCH NO: 33 | 23551A42B3, 23551A4275, 24555A4207")
    print("Starting FastAPI production server at http://127.0.0.1:8000")
    print("==================================================================")
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
