import os
import uvicorn

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    reload = os.environ.get("RELOAD", "false").lower() == "true"
    print("==================================================================")
    print("AUTOMATED MULTI-LESION SEGMENTATION & RETINAL ABNORMALITY DETECTION")
    print("Deep Learning Decision Support Platform (GhostNet, ConvNeXt, U-Net)")
    print(f"Starting FastAPI production server at http://{host}:{port}")
    print("==================================================================")
    uvicorn.run("app.main:app", host=host, port=port, reload=reload)
