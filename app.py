import cv2
import numpy as np
import time
import threading
import queue
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import base64
import logging

app = Flask(__name__)
socketio = SocketIO(app)
logging.basicConfig(level=logging.INFO)

class HandTracker:
    def __init__(self):
        self.initialize_mediapipe()
        self.initialize_state()

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

    def initialize_state(self):
        self.lock = threading.Lock()
        self.frame_counter = 0

    def convert_to_normalized_landmarks(self, hand_landmarks):
        landmark_list = landmark_pb2.NormalizedLandmarkList()
        for landmark in hand_landmarks:
            new_landmark = landmark_list.landmark.add()
            new_landmark.x = landmark.x
            new_landmark.y = landmark.y
            new_landmark.z = landmark.z
        return landmark_list

    def draw_landmarks_and_info(self, frame, result):
        try:
            if not result.hand_landmarks:
                return frame

            annotated_frame = frame.copy()
            frame_height, frame_width = annotated_frame.shape[:2]

            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame_width, frame_height), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.2, annotated_frame, 0.8, 0, annotated_frame)

            for idx, hand_landmarks in enumerate(result.hand_landmarks):
                landmark_list = self.convert_to_normalized_landmarks(hand_landmarks)
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    landmark_list,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=3, circle_radius=6),
                    self.mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=4, circle_radius=3))

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

            cv2.putText(annotated_frame, f"Frame: {self.frame_counter}",
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            return annotated_frame
        except Exception as e:
            logging.error(f"Error drawing landmarks: {e}")
            return frame

    def process_frame(self, frame_data):
        try:
            # Decode base64 image data
            img_data = base64.b64decode(frame_data.split(',')[1])
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            self.frame_counter += 1

            # Process with MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            self.recognizer.recognize_async(mp_image, self.frame_counter)

            # Get results
            if not self.result_queue.empty():
                result, _, _ = self.result_queue.get()
                processed_frame = self.draw_landmarks_and_info(frame, result)
            else:
                processed_frame = frame

            # Encode back to JPEG
            _, buffer = cv2.imencode('.jpg', processed_frame)
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            return None

    def __del__(self):
        if hasattr(self, 'recognizer'):
            self.recognizer.close()
        logging.info("Resources released")

# Initialize tracker
tracker = HandTracker()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('frame')
def handle_frame(data):
    processed_data = tracker.process_frame(data)
    if processed_data:
        emit('processed_frame', processed_data)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
