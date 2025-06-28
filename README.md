# PPE-Kit Detection System

AI-powered Personal Protective Equipment (PPE) detection and compliance monitoring system built with Flask, Ultralytics YOLO, and OpenCV.

## 🚀 Features
- Real-time detection of PPE (helmet, gloves, vest, boots, goggles, etc.) in images and videos
- Supports image and video uploads (JPG, PNG, MP4, AVI, MOV)
- Instant visual feedback with annotated results
- Detailed compliance analysis (detected and missing PPE)
- Modern, responsive web interface
- Built with Ultralytics YOLOv8 and OpenCV

## 🖥️ Demo
<img width="464" alt="Screenshot 2025-06-28 221207" src="https://github.com/user-attachments/assets/a9a2b8e0-28e4-4f39-b96d-2760d5a46d23" />
<img width="646" alt="Screenshot 2025-06-28 222355" src="https://github.com/user-attachments/assets/a72b930b-9ea5-41b9-b452-05303743313b" />


## 📂 Project Structure
```
ppe_flask_app/
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html          # Web interface
├── static/
│   ├── uploads/            # Uploaded files
│   └── results/            # Output results
├── Trained Model/
│   └── best.pt             # YOLOv8 trained model (required)
└── runs/
```

## ⚙️ Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/ShrikantWadkar14/ppe_flask_app.git
   cd ppe_flask_app
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download the trained YOLOv8 model:**
   - Place your `best.pt` model file in the `Trained Model/` directory.
   - (You can train your own model using Ultralytics YOLO or request the model from the author.)

4. **(Optional) Install FFmpeg:**
   - For video processing, [FFmpeg](https://ffmpeg.org/download.html) must be installed and accessible in your system PATH.

## ▶️ Usage
1. **Run the Flask app:**
   ```bash
   python app.py
   ```
2. **Open your browser and go to:**
   [http://127.0.0.1:5000/](http://127.0.0.1:5000/)
3. **Upload an image or video** to analyze PPE compliance. Results and analysis will be displayed instantly.

## 📝 Requirements
- Python 3.8+
- Flask
- Ultralytics
- OpenCV-Python
- FFmpeg (for video re-encoding)

Install all Python dependencies with:
```bash
pip install -r requirements.txt
```

## 📢 Credits
- Developed by Shrikant Wadkar
- shrikantwadkar100@gmail.com
- (For best.pt file contact on email)

## 📄 License
This project is for educational and research purposes. Contact the author for commercial use. 
