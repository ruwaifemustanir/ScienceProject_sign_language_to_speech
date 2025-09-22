import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
from sklearn.neighbors import KNeighborsClassifier
from gesture_manager import GestureManager

class GestureRecognitionApp:
    """
    A real-time hand gesture recognition application using OpenCV and MediaPipe.
    Allows for interactive training and recognition of custom gestures.
    This version highlights the individual landmark points for better visualization.
    """
    def __init__(self):
        # --- App Configuration ---
        self.SAMPLES_TO_COLLECT = 50 # Number of samples per training session
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX
        self.LANDMARK_HIGHLIGHT_COLOR = (0, 255, 0) # Bright green for landmarks

        # --- Initialization ---
        self.gesture_manager = GestureManager('gestures.json')
        self.tts_engine = pyttsx3.init()
        self.cap = cv2.VideoCapture(0)

        # --- MediaPipe Hands Setup ---
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # --- ML Model ---
        self.knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
        self.is_model_trained = False
        self.train_model() # Initial training on saved gestures

        # --- App State ---
        self.mode = 'recognition' # Modes: 'recognition', 'training', 'add_gesture'
        self.current_gesture_name = ""
        self.training_samples_collected = 0
        self.last_spoken_gesture = None
        self.recognized_gesture_name = ""

    def train_model(self):
        """Trains the k-NN classifier with gesture data from the GestureManager."""
        X, y = self.gesture_manager.get_training_data()
        if X and y and len(X) > 0:
            print(f"Training model with {len(X)} samples across {len(set(y))} gestures.")
            self.knn.fit(X, y)
            self.is_model_trained = True
            print("Model trained successfully.")
        else:
            self.is_model_trained = False
            print("Not enough data to train. Please add and train gestures.")

    def normalize_landmarks(self, hand_landmarks):
        """
        Normalizes hand landmarks to make them invariant to hand position,
        scale, and rotation within the frame.
        """
        coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        origin = coords[0]
        coords_translated = coords - origin
        scale_vector = coords_translated[9]
        scale = np.linalg.norm(scale_vector)
        if scale < 1e-6: return None
        coords_scaled = coords_translated / scale
        return coords_scaled.flatten()

    def speak(self, text):
        """Uses the text-to-speech engine to announce text."""
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")

    def process_frame(self, frame):
        """Processes a single video frame for hand detection and recognition."""
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        self.recognized_gesture_name = "" # Reset on each frame without a hand
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the hand connections first
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())

                # --- NEW: Highlight landmarks by drawing circles on top ---
                for landmark in hand_landmarks.landmark:
                    # Convert normalized coordinates to pixel coordinates
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 5, self.LANDMARK_HIGHLIGHT_COLOR, cv2.FILLED)
                # --- End of new code ---

                normalized_landmarks = self.normalize_landmarks(hand_landmarks)
                if normalized_landmarks is not None:
                    if self.mode == 'recognition' and self.is_model_trained:
                        self.handle_recognition(normalized_landmarks)
                    elif self.mode == 'training':
                        self.handle_training(normalized_landmarks)
        else:
             self.last_spoken_gesture = None # Reset if no hand is detected

    def handle_recognition(self, landmarks):
        """Predicts gesture from landmarks and speaks the result."""
        prediction = self.knn.predict([landmarks])
        label = prediction[0]
        labels_to_names = {g['label']: name for name, g in self.gesture_manager.get_all_gestures().items()}
        self.recognized_gesture_name = labels_to_names.get(label, "Unknown")

        if self.recognized_gesture_name != "Unknown" and self.recognized_gesture_name != self.last_spoken_gesture:
            self.speak(self.recognized_gesture_name)
            self.last_spoken_gesture = self.recognized_gesture_name

    def handle_training(self, landmarks):
        """Collects samples for training a gesture."""
        if self.training_samples_collected < self.SAMPLES_TO_COLLECT:
            self.gesture_manager.add_gesture_sample(self.current_gesture_name, landmarks)
            self.training_samples_collected += 1
        else:
            self.gesture_manager.save_gestures()
            self.train_model()
            self.speak(f"Finished training for {self.current_gesture_name}")
            self.mode = 'recognition'
            self.current_gesture_name = ""
            self.training_samples_collected = 0

    def draw_ui(self, frame):
        """Draws UI elements and instructions on the video frame."""
        h, w, _ = frame.shape
        
        info_text = f"Mode: {self.mode.upper()} | 'q' to quit"
        cv2.putText(frame, info_text, (20, 40), self.FONT, 0.8, (0, 0, 0), 2)

        if self.mode == 'recognition':
            cv2.putText(frame, f"Recognized: {self.recognized_gesture_name}", (20, 80), self.FONT, 1, (0, 255, 0), 2)
            cv2.putText(frame, "'a' to Add Gesture | 't' to Train Existing", (20, h - 20), self.FONT, 0.7, (255, 255, 255), 2)
        
        elif self.mode == 'add_gesture':
            cv2.putText(frame, f"New Gesture: {self.current_gesture_name}", (20, 80), self.FONT, 1, (255, 255, 0), 2)
            cv2.putText(frame, "Type name, ENTER to confirm, ESC to cancel", (20, h - 20), self.FONT, 0.7, (255, 255, 255), 2)
        
        elif self.mode == 'training':
            cv2.putText(frame, f"Training: '{self.current_gesture_name}'", (20, 80), self.FONT, 1, (0, 255, 255), 2)
            progress = self.training_samples_collected / self.SAMPLES_TO_COLLECT
            bar_w = int(w * 0.6)
            cv2.rectangle(frame, (20, 100), (20 + bar_w, 130), (100, 100, 100), -1)
            cv2.rectangle(frame, (20, 100), (int(20 + bar_w * progress), 130), (0, 255, 255), -1)
            cv2.putText(frame, f"{self.training_samples_collected}/{self.SAMPLES_TO_COLLECT}", (25 + bar_w, 125), self.FONT, 0.7, (255, 255, 255), 2)

    def run(self):
        """Main application loop."""
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            self.process_frame(frame)
            self.draw_ui(frame)
            cv2.imshow('Hand Gesture Recognition', frame)

            key = cv2.waitKey(5) & 0xFF
            
            if key == ord('q'): break
            elif self.mode == 'recognition':
                if key == ord('a'): self.mode = 'add_gesture'; self.current_gesture_name = ""
                elif key == ord('t'): self.prompt_for_training()
            elif self.mode == 'add_gesture':
                self.handle_add_gesture_input(key)

        self.cleanup()

    def prompt_for_training(self):
        """Handles console input for selecting a gesture to retrain."""
        print("\nAvailable gestures:", self.gesture_manager.get_gesture_names())
        name_to_train = input("Enter the gesture name you want to train more: ")
        if name_to_train in self.gesture_manager.get_gesture_names():
            self.current_gesture_name = name_to_train
            self.mode = 'training'
            self.training_samples_collected = 0
            self.speak(f"Get ready to train {self.current_gesture_name}")
            cv2.waitKey(2000)
        else:
            print("Gesture not found.")

    def handle_add_gesture_input(self, key):
        """Handles keyboard input for naming a new gesture."""
        if key == 27: # ESC
            self.mode = 'recognition'; self.current_gesture_name = ""
        elif key == 13: # ENTER
            if self.current_gesture_name:
                self.mode = 'training'; self.training_samples_collected = 0
                self.speak(f"Get ready to train {self.current_gesture_name}")
                cv2.waitKey(2000)
        elif key == 8: # BACKSPACE
            self.current_gesture_name = self.current_gesture_name[:-1]
        elif 32 <= key <= 126: # Printable characters
            self.current_gesture_name += chr(key)

    def cleanup(self):
        """Releases resources before exiting."""
        self.cap.release()
        cv2.destroyAllWindows()
        self.gesture_manager.save_gestures()
        print("Application closed.")

if __name__ == '__main__':
    app = GestureRecognitionApp()
    app.run()


