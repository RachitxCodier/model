from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import io
from PIL import Image
import uvicorn
import socket

app = FastAPI(title="SmartCart-AI Product Detection API")

# Enable CORS for Flutter app (allow all for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the YOLOv8 model
try:
    model = YOLO("best.pt")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.get("/")
async def root():
    return {"message": "SmartCart-AI Backend is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server.")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Run inference
        results = model.predict(image)
        
        # Get predictions
        # For classification, results[0].probs contains probabilities
        if hasattr(results[0], 'probs'):
            top1_index = results[0].probs.top1
            top1_label = results[0].names[top1_index]
            top1_conf = float(results[0].probs.top1conf)
            
            return {
                "success": True,
                "label": top1_label,
                "confidence": top1_conf,
                "all_predictions": [
                    {"label": results[0].names[i], "confidence": float(results[0].probs.data[i])}
                    for i in range(len(results[0].names))
                ]
            }
        else:
            return {"success": False, "message": "No classification data found in model output."}

    except Exception as e:
        return {"success": False, "error": str(e)}

def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

if __name__ == "__main__":
    ip_addr = get_ip()
    print(f"\n{'='*50}")
    print(f"Server starting on IP: {ip_addr}")
    print(f"Flutter IP: {ip_addr}:8000/predict")
    print(f"{'='*50}\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
