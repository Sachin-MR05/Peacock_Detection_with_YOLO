from flask import Flask, request, jsonify, send_from_directory, Response
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import os
import uuid
import threading
import time
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

model = YOLO("model/best_peacock_model_v3.pt")  # your baseline model
VIDEO_INPUT_DIR = "./data/videos_in"
VIDEO_OUTPUT_DIR = "./data/videos_out"
STREAM_SESSIONS = {}
TRACKER_SESSIONS = {}  # Store tracker state for each device camera session

os.makedirs(VIDEO_INPUT_DIR, exist_ok=True)
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)

@app.route("/")
def home():
    with open('index.html', 'r', encoding='utf-8') as f:
        return f.read()

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["image"]
        
        # Get confidence threshold from frontend (default: 0.5)
        conf_threshold = request.form.get('confidence', 0.5, type=float)
        
        # Save with the original uploaded filename
        filepath = f"./data/{file.filename}"
        file.save(filepath)
        print(f"Image saved to: {filepath}")

        # Use configurable confidence threshold with peacock class filter
        results = model(filepath, conf=conf_threshold, classes=[0])
        print(f"Model inference complete")

        # Read the image
        img = cv2.imread(filepath)
        
        if img is None:
            print(f"ERROR: Failed to read image from {filepath}")
            return jsonify({
                "success": False,
                "error": f"Failed to read image file: {filepath}"
            }), 400
        
        print(f"Image shape: {img.shape}")
        
        detections = []
        for r in results:
            print(f"Total boxes found: {len(r.boxes)}")
            for box in r.boxes:
                # Only process peacock class (class 0)
                if int(box.cls) != 0:
                    continue
                    
                confidence = float(box.conf)
                bbox = box.xyxy.tolist()[0] if len(box.xyxy.shape) > 1 else box.xyxy.tolist()
                print(f"Detection - Class: {int(box.cls)}, Confidence: {confidence}, BBox: {bbox}")
                
                # Draw bounding box on image
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Put confidence text
                label = f"Peacock: {confidence:.2f}"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                detections.append({
                    "class": int(box.cls),
                    "confidence": float(confidence),
                    "bbox": [float(x) for x in bbox]
                })

        # Convert image to base64
        success, buffer = cv2.imencode('.jpg', img)
        if not success:
            print(f"ERROR: Failed to encode image to JPEG")
            return jsonify({
                "success": False,
                "error": "Failed to encode image"
            }), 400
        
        image_base64 = base64.b64encode(buffer).decode()
        print(f"Image encoded successfully, size: {len(image_base64)} bytes")
        print(f"Sending {len(detections)} detections to frontend")

        return jsonify({
            "success": True,
            "image": image_base64,
            "detections": detections,
            "count": len(detections),
            "threshold_used": conf_threshold
        })
    
    except Exception as e:
        print(f"ERROR in predict: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/track-video", methods=["POST"])
def track_video():
    try:
        file = request.files.get("video")
        if file is None or not file.filename:
            return jsonify({"success": False, "error": "No video file uploaded"}), 400

        conf_threshold = request.form.get("confidence", 0.5, type=float)

        safe_name = secure_filename(file.filename)
        input_name = f"{uuid.uuid4().hex}_{safe_name}"
        input_path = os.path.join(VIDEO_INPUT_DIR, input_name)
        file.save(input_path)
        print(f"Video saved to: {input_path}")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return jsonify({"success": False, "error": "Failed to open uploaded video"}), 400

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0

        output_name = f"tracked_{uuid.uuid4().hex}.mp4"
        output_path = os.path.join(VIDEO_OUTPUT_DIR, output_name)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not writer.isOpened():
            cap.release()
            return jsonify({"success": False, "error": "Failed to initialize output video writer"}), 500

        frame_idx = 0
        unique_track_ids = set()
        total_tracked_instances = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(
                frame,
                conf=conf_threshold,
                persist=True,
                verbose=False,
                tracker="bytetrack.yaml"
            )

            result = results[0]
            annotated = result.plot()
            writer.write(annotated)

            if result.boxes is not None and result.boxes.id is not None:
                ids = result.boxes.id.int().cpu().tolist()
                total_tracked_instances += len(ids)
                for track_id in ids:
                    unique_track_ids.add(int(track_id))

            frame_idx += 1

        cap.release()
        writer.release()

        print(
            f"Video tracking complete. Frames: {frame_idx}, "
            f"Unique tracks: {len(unique_track_ids)}"
        )

        return jsonify({
            "success": True,
            "video_url": f"/videos/{output_name}",
            "frames_processed": frame_idx,
            "unique_tracks": len(unique_track_ids),
            "total_tracked_instances": total_tracked_instances,
            "threshold_used": conf_threshold
        })

    except Exception as e:
        print(f"ERROR in track_video: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/prepare-video-stream", methods=["POST"])
def prepare_video_stream():
    try:
        file = request.files.get("video")
        if file is None or not file.filename:
            return jsonify({"success": False, "error": "No video file uploaded"}), 400

        conf_threshold = request.form.get("confidence", 0.2, type=float)
        safe_name = secure_filename(file.filename)
        session_id = uuid.uuid4().hex
        input_name = f"{session_id}_{safe_name}"
        input_path = os.path.join(VIDEO_INPUT_DIR, input_name)
        file.save(input_path)

        STREAM_SESSIONS[session_id] = {
            "path": input_path,
            "confidence": conf_threshold,
            "created_at": time.time(),
        }

        print(f"Prepared live stream session: {session_id} -> {input_path}")

        return jsonify({
            "success": True,
            "stream_url": f"/video-stream/{session_id}",
            "session_id": session_id,
            "threshold_used": conf_threshold,
        })

    except Exception as e:
        print(f"ERROR in prepare_video_stream: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


def generate_video_stream(video_path, conf_threshold):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video {video_path}")
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_fps = src_fps if src_fps and src_fps > 0 else 30.0
    
    # Get video resolution to detect 4K
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    is_4k = frame_width >= 3840  # UHD/4K detection
    print(f"Video resolution: {frame_width}x? - {'4K detected' if is_4k else 'Standard resolution'}")

    frame_idx = 0
    detection_count = 0
    # Keep inference rate around 8 FPS and reuse latest boxes between inference frames.
    last_boxes = []
    
    # For 4K: aggressive settings for small objects
    if is_4k:
        # For 4K with tiny peacocks, we need maximum fidelity
        video_conf = 0.05  # Very low threshold for small objects
        inference_imgsz = 1536  # Maximum inference size for detail
        downscale_width = 2560  # Minimal downscale - keep ~2/3 resolution
        # Increase inference frequency for 4K
        inference_stride = max(1, int(round(src_fps / 12.0)))  # 12 FPS inference instead of 8
        print(f"4K AGGRESSIVE mode: conf={video_conf}, imgsz={inference_imgsz}, keep width={downscale_width}, stride={inference_stride}")
    else:
        video_conf = max(0.15, float(conf_threshold))
        inference_imgsz = 960
        downscale_width = 1280
        inference_stride = max(1, int(round(src_fps / 8.0)))
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Downscale very large frames for much faster inference and smoother streaming.
            h, w = frame.shape[:2]
            if w > downscale_width:
                scale = downscale_width / w
                infer_frame = cv2.resize(frame, (downscale_width, int(h * scale)), interpolation=cv2.INTER_AREA)
            else:
                infer_frame = frame

            annotated = infer_frame.copy()

            if frame_idx % inference_stride == 0:
                results = model.predict(
                    infer_frame,
                    conf=video_conf,
                    verbose=False,
                    imgsz=inference_imgsz,
                    classes=[0]
                )

                result = results[0]
                current_boxes = []
                if result.boxes is not None:
                    for box in result.boxes:
                        coords = box.xyxy[0].tolist()
                        conf = float(box.conf[0]) if hasattr(box.conf, "__len__") else float(box.conf)
                        current_boxes.append((coords, conf))

                last_boxes = current_boxes
                num_boxes = len(current_boxes)
                detection_count += num_boxes
                if frame_idx % 10 == 0:
                    print(f"Stream frame {frame_idx}: {num_boxes} boxes detected @ conf {video_conf:.2f}")

            for coords, conf in last_boxes:
                x1, y1, x2, y2 = map(int, coords)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    f"Peacock {conf:.2f}",
                    (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            display_w = 960
            h, w = annotated.shape[:2]
            if w > display_w:
                display_h = int(h * (display_w / w))
                annotated = cv2.resize(annotated, (display_w, display_h), interpolation=cv2.INTER_AREA)

            ok, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ok:
                continue

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
            )
            frame_idx += 1
    finally:
        cap.release()
        print(f"Live stream finished: {frame_idx} frames, {detection_count} total detections")


def generate_webcam_stream(conf_threshold):
    cap = None
    for idx in range(3):
        tmp = cv2.VideoCapture(idx)
        if tmp.isOpened():
            cap = tmp
            print(f"Using webcam index {idx}")
            break
        tmp.release()

    if cap is None:
        print("ERROR: Cannot open any webcam (indices 0-2)")
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_fps = src_fps if src_fps and src_fps > 0 else 30.0

    frame_idx = 0
    detection_count = 0
    inference_stride = max(1, int(round(src_fps / 8.0)))
    last_boxes = []
    video_conf = max(0.15, float(conf_threshold))

    print(f"Starting webcam stream at ~{src_fps:.1f} FPS, stride {inference_stride}, conf {video_conf:.2f}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            if w > 1280:
                scale = 1280.0 / w
                infer_frame = cv2.resize(frame, (1280, int(h * scale)), interpolation=cv2.INTER_AREA)
            else:
                infer_frame = frame

            annotated = infer_frame.copy()

            if frame_idx % inference_stride == 0:
                results = model.predict(
                    infer_frame,
                    conf=video_conf,
                    verbose=False,
                    imgsz=960,
                    classes=[0]
                )

                result = results[0]
                current_boxes = []
                if result.boxes is not None:
                    for box in result.boxes:
                        coords = box.xyxy[0].tolist()
                        conf = float(box.conf[0]) if hasattr(box.conf, "__len__") else float(box.conf)
                        current_boxes.append((coords, conf))

                last_boxes = current_boxes
                num_boxes = len(current_boxes)
                detection_count += num_boxes
                if frame_idx % 10 == 0:
                    print(f"Webcam frame {frame_idx}: {num_boxes} boxes detected @ conf {video_conf:.2f}")

            for coords, conf in last_boxes:
                x1, y1, x2, y2 = map(int, coords)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    f"Peacock {conf:.2f}",
                    (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            display_w = 960
            h, w = annotated.shape[:2]
            if w > display_w:
                display_h = int(h * (display_w / w))
                annotated = cv2.resize(annotated, (display_w, display_h), interpolation=cv2.INTER_AREA)

            ok, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ok:
                continue

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
            )
            frame_idx += 1
    finally:
        cap.release()
        print(f"Webcam stream finished: {frame_idx} frames, {detection_count} total detections")


@app.route("/video-stream/<session_id>")
def video_stream(session_id):
    session = STREAM_SESSIONS.get(session_id)
    if not session:
        return jsonify({"success": False, "error": "Invalid or expired stream session"}), 404

    return Response(
        generate_video_stream(session["path"], session["confidence"]),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route("/webcam-stream")
def webcam_stream():
    conf_threshold = request.args.get("confidence", 0.2, type=float)
    return Response(
        generate_webcam_stream(conf_threshold),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route("/videos/<path:filename>")
def serve_video(filename):
    return send_from_directory(VIDEO_OUTPUT_DIR, filename, mimetype="video/mp4")


@app.route("/track-frame", methods=["POST"])
def track_frame():
    """
    Real-time frame tracking endpoint for device camera.
    Maintains persistent tracker state per session for continuous object tracking.
    """
    try:
        image_file = request.files.get("image")
        if not image_file:
            return jsonify({"success": False, "error": "No image provided"}), 400

        conf_threshold = request.form.get("confidence", 0.2, type=float)
        session_id = request.form.get("session_id", "default")

        # Read frame from upload
        import io
        image_bytes = image_file.read()
        nparr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        if nparr is None:
            return jsonify({"success": False, "error": "Failed to decode image"}), 400

        # Downscale if too large for faster tracking
        h, w = nparr.shape[:2]
        if w > 1280:
            scale = 1280.0 / w
            nparr = cv2.resize(nparr, (1280, int(h * scale)), interpolation=cv2.INTER_AREA)

        # Detect if input is high resolution (4K/UHD)
        h, w = nparr.shape[:2]
        is_4k = w >= 3840
        
        # Adaptive settings for 4K videos - aggressive for tiny objects
        if is_4k:
            track_imgsz = 1536  # Maximum inference size for 4K
            track_conf = 0.05  # Very low threshold for small objects
            print(f"4K frame tracking: imgsz={track_imgsz}, conf={track_conf}")
        else:
            track_imgsz = 832  # Good balance for standard resolution
            track_conf = max(0.15, conf_threshold)
        
        # Track with persistent state per session
        results = model.track(
            nparr,
            conf=track_conf,
            persist=True,
            verbose=False,
            imgsz=track_imgsz,
            tracker="bytetrack.yaml",
            classes=[0]  # Only peacock class
        )

        result = results[0]
        annotated = result.plot()

        # Extract detections with track IDs
        detections = []
        if result.boxes is not None:
            print(f"✅ Found {len(result.boxes)} peacock(s) in frame")
            for i, box in enumerate(result.boxes):
                track_id = int(box.id[0]) if box.id is not None else -1
                confidence = float(box.conf[0]) if hasattr(box.conf, "__len__") else float(box.conf)
                bbox = box.xyxy[0].tolist()
                
                print(f"  Peacock #{i+1}: ID={track_id}, Conf={confidence:.2f}, BBox={[round(x) for x in bbox]}")
                
                detections.append({
                    "track_id": track_id,
                    "class": int(box.cls),
                    "confidence": confidence,
                    "bbox": [float(x) for x in bbox]
                })

        # Encode annotated frame to base64
        success, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not success:
            return jsonify({"success": False, "error": "Failed to encode frame"}), 500

        image_base64 = base64.b64encode(buffer).decode()

        return jsonify({
            "success": True,
            "image": image_base64,
            "detections": detections,
            "count": len(detections),
            "threshold_used": conf_threshold
        })

    except Exception as e:
        print(f"ERROR in track_frame: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    # Use 0.0.0.0 so other devices on the same network can access this server
    app.run(debug=False, host='0.0.0.0', port=5000)