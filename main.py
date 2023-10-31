from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.boxlayout import BoxLayout

from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.graphics import Color, Line, Ellipse
from kivy.core.window import Window
from kivy.animation import Animation
import librosa
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from scipy.io import wavfile
import sounddevice as sd


# Set the window size
Window.size = (360, 640)  # width x height

# Optional: Disable window resizing
Window.resizable = True

class Chord_classifier():
    def __init__(self, model, encoder):
        self.model = model
        self.label_encoder = encoder
    
    def get_notes_for_chord(self, chord):
        chord_notes_mapping = {
        'Am': ['A', 'C', 'E'],
        'Bb': ['Bb', 'D', 'F'],
        'Bdim': ['B', 'D', 'F'],
        'C': ['C', 'E', 'G'],
        'Dm': ['D', 'F', 'A'],
        'Em': ['E', 'G', 'B'],
        'F': ['F', 'A', 'C'],
        'G': ['G', 'B', 'D']
        # Add more chord names and their corresponding notes here
        }
        if chord in chord_notes_mapping:
            # Get the list of notes corresponding to the chord
            chord_notes = chord_notes_mapping[chord]
            
            # Return the first three notes from the list (or fewer if there are less than three)
            return chord_notes[:3]
        else:
            # Handle the case when the chord is not in the dictionary
            return []

    def extract_features(self, audio_file):
        chord, fs = librosa.load(audio_file, sr=None)
        chord_emphasized = librosa.effects.preemphasis(chord, coef=0.97)
        chromagram = librosa.feature.chroma_cens(y=chord_emphasized, sr=fs, hop_length=512)
        pitch_sum = chromagram.sum(axis=1)
        return pitch_sum

    def predict_new_chord(self, audio_file_path, model, label_encoder):
        # Extract features from the new audio file
        feature_vector = self.extract_features(audio_file_path)

        # Reshape the feature vector to match the model's input shape
        feature_vector = feature_vector.reshape(1, -1)

        # Use the trained model to make a prediction
        predicted_label = self.model.predict(feature_vector)

        # Decode the predicted label to get the chord name
        predicted_chord = self.label_encoder.inverse_transform(predicted_label)

        return predicted_chord[0]

class ChordCircle(Widget):
    center_x = Window.width * 0.23
    center_y = Window.height - 300
    radius = min(Window.width, Window.height) * 0.25 - 20  # Deduct 20 to account for the small circles


    def __init__(self, chord, **kwargs):
        super(ChordCircle, self).__init__(**kwargs)
        self.chord = chord or []
        self.draw_chr_circle()

    def update_chord(self, new_chord):
        self.chord = new_chord
        self.canvas.clear()
        self.draw_chr_circle()

    def draw_chr_circle(self):
        # Define the notes and their positions
        self.canvas.clear()
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        num_notes = len(notes)
        theta = np.linspace(0, 2*np.pi, num_notes, endpoint=False)
        x = np.cos(theta) * self.radius + self.center_x
        y = np.sin(theta) * self.radius + self.center_y

        with self.canvas:
            Color(0.5, 0.5, 0.5)
            Line(circle=(self.center_x, self.center_y, self.radius))
            for i, note in enumerate(notes):
                Line(points=[x[i], y[i], x[(i+1) % num_notes], y[(i+1) % num_notes]])

            Color(1, 0, 0)
            for i in range(len(self.chord)):
                start_note = self.chord[i]
                end_note = self.chord[(i+1) % len(self.chord)]
                start_idx = notes.index(start_note)
                end_idx = notes.index(end_note)
                Line(points=[x[start_idx], y[start_idx], x[end_idx], y[end_idx]])

            for i, note in enumerate(notes):
                # Drawing small circles and text labels for notes
                Color(1, 1, 1)
                Ellipse(pos=(x[i]-10, y[i]-10), size=(20, 20))

                Color(0, 0, 0)
                
                # Add labels here if needed using Kivy's Label widget
                note_label = Label(text=note, center=(round(x[i]), round(y[i])), font_size=12, color=(0, 0, 0, 1))
                self.add_widget(note_label)

class MyApp(App):
    sample_rate = 44100
    chord = ['C','E','G']
    def build(self):
        # Load the trained model and label encoder
        self.model = joblib.load('chord_identifier.pkl')
        self.label_encoder = joblib.load('label_encoder.pkl')
        self.classifier = Chord_classifier(self.model, self.label_encoder)
        
        # Create a layout to hold both the ChordCircle and the record button
        self.main_layout = BoxLayout(orientation='vertical', spacing=10, padding=20, size_hint=(None, None))
        self.main_layout.bind(minimum_size=self.main_layout.setter('size'))
        
        # Add a record button
        record_button = Button(text="Record Chord", size_hint=(None, None), size=(100, 50))
        record_button.bind(on_press=self.record_chord)
        
        # Create a ChordCircle widget with a default chord and add it to the layout
        
        self.chord_circle = ChordCircle(chord=self.chord, size_hint=(1, 1))
        
        self.main_layout.add_widget(self.chord_circle)
        self.main_layout.add_widget(record_button)
        
        self.recording = False  # Flag to indicate whether recording is in progress
        self.recorded_audio = None  # To store the recorded audio data

        return self.main_layout

    def record_chord(self, instance):
        if not self.recording:
            # Start recording
            self.recording = True
            instance.text = "Stop Recording"

            # Set the sample rate and duration for recording
            duration = 5  # Record for 5 seconds (you can adjust this too)

            # Start recording audio
            self.recorded_audio = sd.rec(int(self.sample_rate * duration), samplerate=self.sample_rate, channels=1)
        else:
            # Stop recording
            self.recording = False
            instance.text = "Record Chord"
            
            # Save the recorded audio to a WAV file
            if self.recorded_audio is not None:
                wavfile.write('recorded_chord.wav', self.sample_rate, self.recorded_audio)
                chord_from_audio = self.classifier.predict_new_chord('recorded_chord.wav',self.model, self.label_encoder)
                notes = self.classifier.get_notes_for_chord(chord_from_audio)
                self.chord = notes
                self.chord_circle.update_chord(self.chord)

                

if __name__ == '__main__':
    MyApp().run()