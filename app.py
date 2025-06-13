import cv2
import numpy as np
import time
import threading
import queue
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from flask import Flask, Response, render_template
import logging
import datetime

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

class HandTracker:
    def __init__(self):
        self.initialize_mediapipe()
        self.setup_camera()
        self.initialize_state()
        self.last_frame_time = time.time()

    def initialize_mediapipe(self):
        model_path = 'gesture_recognizer.task'
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=2,
            result_callback=self.process_result
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)
        self.result_queue = queue.Queue()
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        logging.info("MediaPipe initialized")

    def process_result(self, result, output_image, timestamp_ms):
        self.result_queue.put((result, output_image, timestamp_ms))

    def setup_camera(self):
        self.cap = None
        self.camera_active = False
        
        # Try different camera indices and backends
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]
        for backend in backends:
            for i in range(0, 3):
                try:
                    self.cap = cv2.VideoCapture(i, backend)
                    if self.cap.isOpened():
                        # Test if we can actually get a frame
                        ret, frame = self.cap.read()
                        if ret:
                            self.camera_active = True
                            logging.info(f"Camera found at index {i} with backend {backend}")
                            break
                        else:
                            self.cap.release()
                except:
                    pass
            if self.camera_active:
                break
        
        if not self.camera_active:
            # Last resort: try without specifying backend
            try:
                self.cap = cv2.VideoCapture(0)
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        self.camera_active = True
                        logging.info("Camera found with default backend")
            except:
                pass
        
        if not self.camera_active:
            logging.error("Could not open any camera")
            return
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        logging.info(f"Camera resolution: {actual_width}x{actual_height}, FPS: {actual_fps}")

    def initialize_state(self):
        self.stop_event = threading.Event()
        self.frame_queue = queue.Queue(maxsize=3)  # Increased queue size
        self.output_frame = None
        self.lock = threading.Lock()
        self.frame_counter = 0
        self.last_successful_frame_time = time.time()

    def capture_frames(self):
        if not self.camera_active:
            logging.error("Camera not active in capture thread")
            return
            
        logging.info("Capture thread started")
        timestamp = 0
        while not self.stop_event.is_set() and self.camera_active:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    logging.warning("Failed to capture frame")
                    time.sleep(0.05)
                    continue
                
                self.frame_counter += 1
                self.last_successful_frame_time = time.time()
                
                # Add timestamp to frame for debugging
                cv2.putText(frame, f"Frame: {self.frame_counter}", (20, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if self.frame_queue.full():
                    self.frame_queue.get()
                    
                self.frame_queue.put((frame, timestamp))
                timestamp += 1
                time.sleep(0.01)
            except Exception as e:
                logging.error(f"Error in capture thread: {e}")
                time.sleep(0.1)

    def convert_to_normalized_landmarks(self, hand_landmarks):
        landmark_list = landmark_pb2.NormalizedLandmarkList()
        for landmark in hand_landmarks:
            new_landmark = landmark_list.landmark.add()
            new_landmark.x = landmark.x
            new_landmark.y = landmark.y
            new_landmark.z = landmark.z
        return landmark_list

    def process_frames(self):
        logging.info("Process thread started")
        while not self.stop_event.is_set():
            try:
                # Skip processing if queue is empty
                if self.frame_queue.empty():
                    time.sleep(0.01)
                    continue
                    
                frame, timestamp = self.frame_queue.get()
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                self.recognizer.recognize_async(mp_image, timestamp)
                
                # Process results if available
                if not self.result_queue.empty():
                    result, output_image, _ = self.result_queue.get()
                    annotated_frame = self.draw_landmarks_and_info(frame, result)
                    with self.lock:
                        self.output_frame = annotated_frame
                else:
                    # If no results, just use the frame as is
                    with self.lock:
                        self.output_frame = frame
            except Exception as e:
                logging.error(f"Error processing frame: {e}")
                time.sleep(0.1)

    def draw_landmarks_and_info(self, frame, result):
        try:
            if not result.hand_landmarks:
                return frame

            annotated_frame = frame.copy()
            frame_height, frame_width = annotated_frame.shape[:2]
            
            # Draw overlay and landmarks
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame_width, frame_height), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.2, annotated_frame, 0.8, 0, annotated_frame)
            
            for idx, hand_landmarks in enumerate(result.hand_landmarks):
                landmark_list = self.convert_to_normalized_landmarks(hand_landmarks)
                
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    landmark_list,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=3, circle_radius=6),
                    self.mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=4, circle_radius=3))
                
                # Draw bounding box and gesture info
                x_coords = [landmark.x for landmark in hand_landmarks]
                y_coords = [landmark.y for landmark in hand_landmarks]
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)
                
                x1, y1 = int(min_x * frame_width), int(min_y * frame_height)
                x2, y2 = int(max_x * frame_width), int(max_y * frame_height)
                
                for i in range(1, 5):
                    alpha = i/5
                    color = (0, int(255*alpha), 0)
                    cv2.rectangle(annotated_frame, (x1-i, y1-i), (x2+i, y2+i), color, 1)
                
                if idx < len(result.gestures):
                    gesture = result.gestures[idx][0]
                    confidence = gesture.score
                    text = f"{gesture.category_name} ({confidence*100:.1f}%)"
                    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(annotated_frame, 
                                 (x1-5, y1-text_height-15), 
                                 (x1+text_width+5, y1-5), 
                                 (40, 40, 40), -1)
                    
                    text_color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255)
                    cv2.putText(annotated_frame, text, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv2.LINE_AA)

            # Add FPS counter
            fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap else 0
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}",
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                       (0, 255, 255), 2, cv2.LINE_AA)
            
            # Add camera status
            status = "Active" if self.camera_active else "Inactive"
            cv2.putText(annotated_frame, f"Status: {status}",
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       (0, 255, 0) if self.camera_active else (0, 0, 255), 2, cv2.LINE_AA)
            
            return annotated_frame
        except Exception as e:
            logging.error(f"Error drawing landmarks: {e}")
            return frame

    def run(self):
        if not self.camera_active:
            logging.error("Cannot start tracking - no camera available")
            return
            
        capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        process_thread = threading.Thread(target=self.process_frames, daemon=True)
        capture_thread.start()
        process_thread.start()
        logging.info("Tracking started")

    def get_frame(self):
        with self.lock:
            # Check if we have recent frames
            has_recent_frame = time.time() - self.last_successful_frame_time < 2.0
            
            if not self.camera_active or not has_recent_frame:
                # Create a blank frame with error message
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                if not self.camera_active:
                    message = "Camera not available"
                    color = (0, 0, 255)
                else:
                    message = "No frames received recently"
                    color = (0, 165, 255)
                
                cv2.putText(blank, message, (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(blank, f"Last frame: {self.frame_counter}", (50, 280), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
                _, buffer = cv2.imencode('.jpg', blank)
                return buffer.tobytes()
                
            if self.output_frame is None:
                # Create a blank frame with loading message
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "Processing initial frame...", (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                _, buffer = cv2.imencode('.jpg', blank)
                return buffer.tobytes()
                
            try:
                ret, buffer = cv2.imencode('.jpg', self.output_frame)
                if ret:
                    return buffer.tobytes()
                return None
            except:
                # If encoding fails, return a blank frame
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "Frame encoding error", (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                _, buffer = cv2.imencode('.jpg', blank)
                return buffer.tobytes()

    def __del__(self):
        self.stop_event.set()
        if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'recognizer'):
            self.recognizer.close()
        logging.info("Resources released")

# Initialize tracker
tracker = HandTracker()
if tracker.camera_active:
    tracker.run()
else:
    logging.error("Camera initialization failed - running in degraded mode")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = tracker.get_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            time.sleep(0.033)  # ~30 FPS
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera_status')
def camera_status():
    return {
        "status": "active" if tracker.camera_active else "inactive",
        "frame_counter": tracker.frame_counter,
        "last_frame_time": tracker.last_successful_frame_time
    }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False)