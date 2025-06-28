import os
from flask import Flask, render_template, request, send_from_directory
from ultralytics import YOLO
import cv2
import time
import subprocess 

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('static', 'uploads')
RESULT_FOLDER = os.path.join('static', 'results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = YOLO('Trained Model/best.pt')
print("Model class mapping:", model.names)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'}

def get_missing_ppe_classes():
    missing_ppe_indices = []
    for idx, name in model.names.items():
        if name.startswith('no_'):
            missing_ppe_indices.append(idx)
    return missing_ppe_indices

def analyze_detection_results(results):
    if not results or len(results) == 0:
        return {"alert": "No detections found", "status": "warning", "details": []}
    
    result = results[0]
    if result.boxes is None:
        return {"alert": "No objects detected", "status": "warning", "details": []}
    
    detected_classes = []
    missing_ppe_detected = []
    proper_ppe_detected = []
    
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]
        
        detected_classes.append({
            'name': class_name,
            'id': class_id,
            'confidence': confidence
        })
        
        if class_name.startswith('no_'):
            missing_ppe_detected.append(class_name)
        elif class_name in ['helmet', 'gloves', 'vest', 'boots', 'goggles']:
            proper_ppe_detected.append(class_name)
    
    if missing_ppe_detected:
        alert_msg = f"ALERT: Missing PPE detected! ({', '.join(missing_ppe_detected)})"
        status = "danger"
    else:
        alert_msg = "All PPE present and compliant."
        status = "success"
    
    return {
        "alert": alert_msg,
        "status": status,
        "details": detected_classes,
        "missing_ppe": missing_ppe_detected,
        "proper_ppe": proper_ppe_detected
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    alert = None
    alert_status = None
    result_filename = None
    uploaded_filename = None

    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            uploaded_filename = filename
            upload_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(upload_path)
            ext = filename.rsplit('.', 1)[1].lower()

            # ---- IMAGE ----
            if ext in ['jpg', 'jpeg', 'png']:
                results = model.predict(
                    source=upload_path,
                    save=True,
                    project='static',
                    name='results',
                    exist_ok=True
                )
                saved_file = os.path.basename(results[0].path)
                result_filename = saved_file

                analysis = analyze_detection_results(results)
                alert = analysis["alert"]
                alert_status = analysis["status"]

            # ---- VIDEO ----
            elif ext in ['mp4', 'avi', 'mov']:
                cap = cv2.VideoCapture(upload_path)
                if not cap.isOpened():
                    alert = "Error: Could not open uploaded video file."
                    alert_status = "warning"
                    return render_template('index.html', 
                                         alert=alert, 
                                         alert_status=alert_status,
                                         result_filename=None, 
                                         uploaded_filename=uploaded_filename)
                
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

                print(f"Video properties: {width}x{height}, {fps} FPS")

                result_filename = 'result_' + os.path.splitext(filename)[0] + '.mp4'
                result_path = os.path.join(RESULT_FOLDER, result_filename)

                codecs_to_try = ['mp4v', 'avc1', 'H264']
                out = None
                for codec in codecs_to_try:
                    try:
                        fourcc = cv2.VideoWriter_fourcc(*codec)
                        out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))
                        if out.isOpened():
                            print(f"Successfully created video with codec: {codec}")
                            break
                        else:
                            out.release()
                    except:
                        if out:
                            out.release()
                        continue

                if not out or not out.isOpened():
                    result_filename = 'result_' + os.path.splitext(filename)[0] + '.avi'
                    result_path = os.path.join(RESULT_FOLDER, result_filename)
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))
                    print("Falling back to AVI format")

                if not out.isOpened():
                    alert = "Error: Could not create output video file."
                    alert_status = "warning"
                    cap.release()
                    return render_template('index.html', 
                                         alert=alert, 
                                         alert_status=alert_status,
                                         result_filename=None, 
                                         uploaded_filename=uploaded_filename)
                
                print(f"Output video path: {result_path}")
                missing_ppe_indices = get_missing_ppe_classes()
                alert_flag = False
                all_detections = {}
                frame_count = 0
                missing_ppe_details = []

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    results = model(frame, conf=0.25)[0]
                    annotated_frame = results.plot()
                    out.write(annotated_frame)

                    if results.boxes is not None:
                        for box in results.boxes:
                            class_id = int(box.cls[0])
                            class_name = model.names[class_id]
                            confidence = float(box.conf[0])
                            
                            if class_name not in all_detections:
                                all_detections[class_name] = {
                                    'count': 0,
                                    'max_confidence': 0,
                                    'id': class_id
                                }
                            all_detections[class_name]['count'] += 1
                            all_detections[class_name]['max_confidence'] = max(
                                all_detections[class_name]['max_confidence'], 
                                confidence
                            )

                            if class_id in missing_ppe_indices:
                                alert_flag = True
                                if class_name not in missing_ppe_details:
                                    missing_ppe_details.append(class_name)
                                print(f"Missing PPE detected: {class_name} (ID: {class_id})")
                    
                    frame_count += 1

              
                cap.release()
                out.release()
                time.sleep(1)

                #  RE-ENCODE to browser-compatible H.264 format
                fixed_filename = 'fixed_' + result_filename
                fixed_result_path = os.path.join(RESULT_FOLDER, fixed_filename)

                try:
                    subprocess.run([
                        r'C:\ffmpeg\bin\ffmpeg.exe', '-y', '-i', result_path,
                        '-vcodec', 'libx264', '-acodec', 'aac',
                        '-movflags', '+faststart',
                        fixed_result_path
                    ], check=True)
                    os.remove(result_path)
                    result_filename = fixed_filename
                    print(f"✅ Re-encoded video saved as: {fixed_filename}")
                except Exception as e:
                    print(f"❌ FFmpeg re-encoding failed: {e}")


                print(f"Processed {frame_count} frames")
                if os.path.exists(os.path.join(RESULT_FOLDER, result_filename)):
                    alert = (
                        f"ALERT: Missing PPE detected in video! ({', '.join(missing_ppe_details)})"
                        if alert_flag else "All PPE present in video."
                    )
                    alert_status = "danger" if alert_flag else "success"

            else:
                alert = "Unsupported file format."
                alert_status = "warning"

    return render_template('index.html', 
                         alert=alert, 
                         alert_status=alert_status,
                         result_filename=result_filename, 
                         uploaded_filename=uploaded_filename)

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/static/results/<filename>')
def result_file(filename):
    if filename.endswith('.avi'):
        return send_from_directory(RESULT_FOLDER, filename, mimetype='video/x-msvideo')
    elif filename.endswith('.mp4'):
        return send_from_directory(RESULT_FOLDER, filename, mimetype='video/mp4')
    return send_from_directory(RESULT_FOLDER, filename)

@app.route('/static/uploads/<filename>')
def upload_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
