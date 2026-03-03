1. README.md
Save as: README.md

Markdown
# 🛡️ SentinelAI: Real-Time Anomaly Detection

**SentinelAI** is an intelligent surveillance system capable of detecting anomalies (such as violence, accidents, or theft) in real-time video feeds. It utilizes a hybrid **CNN-LSTM architecture** to analyze both spatial features (frames) and temporal patterns (sequences), achieving high accuracy in distinguishing normal activities from suspicious behavior.

This project was engineered to run efficiently on edge devices, leveraging **PyTorch** for model training and **FastAPI** for a responsive deployment interface.

---

## 🚀 Key Features

* **Hybrid Architecture**: Combines **ResNet50 (CNN)** for feature extraction and **LSTM** for sequence modeling to understand temporal context.
* **Real-Time Inference**: Processes video streams in chunks using a sliding window technique to detect anomalies instantly.
* **Edge Optimization**: Designed with inference speed in mind, utilizing optimized tensor operations for low-latency alerts.
* **Interactive Dashboard**: Features a **FastAPI** backend serving a lightweight HTML/JS frontend for live video analysis and confidence scoring.

---

## 🛠️ Tech Stack

* **Deep Learning**: PyTorch, Torchvision
* **Computer Vision**: OpenCV (cv2)
* **Model Architecture**: ResNet50 + LSTM (Long Short-Term Memory)
* **Backend**: FastAPI, Uvicorn
* **Frontend**: HTML5, JavaScript (Fetch API)
* **Deployment**: Docker (Compatible), Hugging Face Spaces

---

## 📂 Project Structure

```text
SentinelAI/
├── dataset/                  # Training data (Normal vs Anomaly)
├── models/                   # Saved model weights (sentinel_ai.pth)
├── app.py                    # FastAPI Backend & Web Dashboard
├── train.py                  # Model Training Script
├── model_arch.py             # CNN-LSTM Architecture Definition
├── dataset_loader.py         # Data Loading & Augmentation Pipeline
├── requirements.txt          # Python Dependencies
└── README.md                 # Project Documentation
⚙️ Installation & Setup
1. Clone the Repository
Bash
git clone [https://github.com/Arnab-Pachal1234/SentinelAI.git](https://github.com/Arnab-Pachal1234/SentinelAI.git)
cd SentinelAI
2. Install Dependencies
Bash
pip install -r requirements.txt
3. Train the Model
If the models/sentinel_ai.pth file is missing, train the model from scratch:

Bash
python train.py
Note: This will train for 20 epochs using the dataset in the dataset/ folder.

4. Run the Dashboard
Start the local server:

Bash
uvicorn app:app --reload
Open your browser and navigate to: https://www.google.com/search?q=http://127.0.0.1:8000

🧠 Model Performance
The model uses a Sliding Window Approach during inference:

The video is split into overlapping 20-frame segments.

Each segment is passed through the CNN-LSTM network.

If the anomaly confidence score exceeds 70% for any segment, the system triggers an alert.

👨‍💻 Author
Arnab PachalB.Tech CSE, NIT DurgapurLinkedIn | GitHub | Portfolio
