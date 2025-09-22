import json
import os
import numpy as np

class GestureManager:
    """
    Manages loading, saving, and handling gesture data from a JSON file.
    This class isolates the data persistence logic from the main application.
    """
    def __init__(self, filepath='gestures.json'):
        """
        Initializes the GestureManager.

        Args:
            filepath (str): The path to the JSON file where gestures are stored.
        """
        self.filepath = filepath
        self.gestures = self.load_gestures()

    def load_gestures(self):
        """
        Loads gesture data from the JSON file.
        If the file doesn't exist or is invalid, it returns an empty dictionary.
        """
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                    # Landmark data is saved as lists, so convert back to numpy arrays
                    for name, gesture_data in data.items():
                        gesture_data['landmarks'] = [np.array(lm) for lm in gesture_data['landmarks']]
                    print("Gestures loaded successfully.")
                    return data
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading gestures file: {e}. Starting with an empty gesture set.")
                return {}
        return {}

    def save_gestures(self):
        """Saves the current gestures to the JSON file."""
        try:
            with open(self.filepath, 'w') as f:
                # Create a serializable copy of the gestures data
                # Numpy arrays must be converted to lists for JSON compatibility
                data_to_save = {}
                for name, gesture_data in self.gestures.items():
                    data_to_save[name] = {
                        'label': gesture_data['label'],
                        'landmarks': [lm.tolist() for lm in gesture_data['landmarks']]
                    }
                json.dump(data_to_save, f, indent=4)
            print("Gestures saved successfully.")
        except IOError as e:
            print(f"Error saving gestures: {e}")

    def add_gesture_sample(self, name, landmarks):
        """
        Adds a new sample of landmarks for a given gesture.
        If the gesture is new, it's added to the dictionary.

        Args:
            name (str): The name of the gesture (e.g., "peace", "hello").
            landmarks (np.array): The normalized landmarks for the gesture sample.
        """
        if name not in self.gestures:
            # Assign a new unique integer label for the new gesture
            new_label = max([g['label'] for g in self.gestures.values()] + [-1]) + 1
            self.gestures[name] = {'label': new_label, 'landmarks': []}
            print(f"Added new gesture: '{name}' with label {new_label}")

        self.gestures[name]['landmarks'].append(landmarks)
        print(f"Added sample to '{name}'. Total samples: {len(self.gestures[name]['landmarks'])}")

    def get_all_gestures(self):
        """Returns the dictionary of all gestures."""
        return self.gestures

    def get_gesture_names(self):
        """Returns a list of all trained gesture names."""
        return list(self.gestures.keys())

    def get_training_data(self):
        """
        Prepares the landmarks (X) and labels (y) for training a machine learning model.

        Returns:
            tuple: A tuple containing (X, y).
                   X is a list of all landmark samples.
                   y is a list of their corresponding integer labels.
                   Returns (None, None) if no gestures are available.
        """
        if not self.gestures:
            return None, None

        X, y = [], []
        for name, gesture_data in self.gestures.items():
            label = gesture_data['label']
            for landmarks in gesture_data['landmarks']:
                X.append(landmarks)
                y.append(label)
        return X, y


