import os
import cv2
import torch
import numpy as np
import shutil
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from torchvision import transforms
from model_arch import CNNLSTM

app = FastAPI(title="SentinelAI")

# 1. Load Model
device = torch.device("cpu") # Use CPU for inference to be safe
model = CNNLSTM()
# Ensure models folder exists and contains the pth file
if os.path.exists("models/sentinel_ai.pth"):
    model.load_state_dict(torch.load("models/sentinel_ai.pth", map_location=device))
    print("✅ Model Loaded Successfully")
else:
    print("⚠️ Warning: Model file not found. Please run train.py first.")

model.to(device)
model.eval()

# 2. Preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

SEQ_LEN = 20
CLASSES = ['Normal', 'Anomaly']

def process_video_chunks(video_path):
    """
    Splits the video into MULTIPLE 20-frame chunks (Sliding Window).
    Returns a batch of chunks: (Num_Chunks, 20, 3, 224, 224)
    """
    cap = cv2.VideoCapture(video_path)
    all_chunks = []
    current_chunk = []
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        
        # Transform immediately
        frame_tensor = transform(frame)
        current_chunk.append(frame_tensor)
        
        # If we have a full chunk (20 frames), save it and start over
        if len(current_chunk) == SEQ_LEN:
            # Convert list of tensors to a single tensor for this chunk
            chunk_tensor = torch.stack(current_chunk)
            all_chunks.append(chunk_tensor)
            current_chunk = [] # Reset for next chunk
            
    cap.release()
    
    # If no full chunks (video too short), we can't process it easily
    if not all_chunks:
        return None

    # Stack all chunks into a batch: (Num_Chunks, 20, 3, 224, 224)
    return torch.stack(all_chunks)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Get ALL chunks from the video
        inputs = process_video_chunks(temp_file)
        
        if inputs is None:
             return {"filename": file.filename, "prediction": "Error", "confidence": "Video too short (needs 20+ frames)"}

        inputs = inputs.to(device)
        
        with torch.no_grad():
            # Pass ALL chunks through the model at once
            # outputs shape: (Num_Chunks, 2)
            outputs = model(inputs) 
            
            # Apply Softmax to get probabilities (0.0 to 1.0)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # Extract the "Anomaly" score (Class 1) for every chunk
            anomaly_scores = probs[:, 1] 
            
            # LOGIC: Find the MAXIMUM anomaly score across the whole video
            # If even one 20-frame chunk looks like violence, flag it.
            max_score = torch.max(anomaly_scores).item()
            
            # Threshold: If > 70% confidence in any chunk, it's Anomaly
            if max_score > 0.70: 
                result = "Anomaly"
                confidence_val = max_score
            else:
                result = "Normal"
                # For Normal confidence, we invert the anomaly score
                confidence_val = 1.0 - max_score

        return {
            "filename": file.filename, 
            "prediction": result, 
            "confidence": f"{confidence_val*100:.2f}%"
        }
    
    except Exception as e:
        return {"error": str(e)}
        
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

# 3. The Web Dashboard (HTML)
@app.get("/", response_class=HTMLResponse)
async def main():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SentinelAI Dashboard</title>
        <style>
            body { font-family: 'Segoe UI', sans-serif; text-align: center; padding: 50px; background-color: #f4f4f9; }
            .container { background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); max-width: 500px; margin: auto; }
            h1 { color: #333; }
            .upload-btn { background-color: #007bff; color: white; padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; margin-top: 20px; }
            .upload-btn:hover { background-color: #0056b3; }
            #result { margin-top: 20px; font-weight: bold; font-size: 20px; }
            .Normal { color: green; }
            .Anomaly { color: red; }
            .loader { border: 4px solid #f3f3f3; border-radius: 50%; border-top: 4px solid #3498db; width: 30px; height: 30px; -webkit-animation: spin 1s linear infinite; animation: spin 1s linear infinite; margin: 20px auto; display: none; }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🛡️ SentinelAI Dashboard</h1>
            <p>Real-Time Anomaly Detection System</p>
            <input type="file" id="videoInput" accept="video/*">
            <button class="upload-btn" onclick="uploadVideo()">Analyze Surveillance Feed</button>
            <div class="loader" id="loader"></div>
            <div id="result"></div>
        </div>

        <script>
            async function uploadVideo() {
                const fileInput = document.getElementById('videoInput');
                const resultDiv = document.getElementById('result');
                const loader = document.getElementById('loader');
                
                if (fileInput.files.length === 0) {
                    alert("Please select a video file!");
                    return;
                }

                resultDiv.innerHTML = "";
                loader.style.display = "block";
                
                const formData = new FormData();
                formData.append("file", fileInput.files[0]);

                try {
                    const response = await fetch('/predict/', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await response.json();
                    loader.style.display = "none";
                    
                    if (data.error) {
                        resultDiv.innerHTML = `<span style="color:red">Error: ${data.error}</span>`;
                    } else {
                        resultDiv.innerHTML = `Result: <span class="${data.prediction}">${data.prediction}</span> <br> Confidence: ${data.confidence}`;
                    }
                } catch (error) {
                    loader.style.display = "none";
                    resultDiv.innerHTML = "❌ Error processing video.";
                    console.error(error);
                }
            }
        </script>
    </body>
    </html>
    """
