from kivymd.app import MDApp
from kivymd.uix.button import MDRoundFlatButton, MDFloatingActionButton
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDIconButton

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.graphics import Color, Line, Ellipse, Rectangle
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
    '''Uses the chord_identifier.pkl model to classify any chord.'''
    def __init__(self, model, encoder):
        self.model = model
        self.label_encoder = encoder
    
    def get_notes_for_chord(self, chord):
        '''takes a chord (C#) and gives you the triad notes in that chord'''
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
        '''the machine-learning model recognizes features found in the chromagram.'''
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

    def __init__(self, chord, overlay_mode=False, **kwargs):
        super(ChordCircle, self).__init__(**kwargs)
        self.chords_history = [chord]
        self.overlay_enabled = overlay_mode
        self.toggle = False
        self.toggled_chord = ""
        self.draw_chr_circle()
    
    def highlight_shape(self, chord):
        self.toggle = True
        self.toggled_chord = chord
        self.canvas.clear()
        self.draw_chr_circle()

    def update_chord(self, new_chord):
        if self.overlay_enabled:
            self.chords_history.append(new_chord)
        else:
            self.chords_history = [new_chord]
        self.canvas.clear()
        self.draw_chr_circle()

    def draw_chr_circle(self):
        # Define the notes and their positions
        notes = ['A', 'A#', 'B', 'C', 'C#', 'D','D#', 'E', 'F', 'F#', 'G', 'G#']
        num_notes = len(notes)
        theta = np.linspace(0, 2*np.pi, num_notes, endpoint=False)
        x = np.cos(theta) * self.radius + self.center_x
        y = np.sin(theta) * self.radius + self.center_y

        with self.canvas:
            Color(0.5, 0.5, 0.5)
            Line(circle=(self.center_x, self.center_y, self.radius))
            for i, note in enumerate(notes):
                Line(points=[x[i], y[i], x[(i+1) % num_notes], y[(i+1) % num_notes]])

            for idx, chord in enumerate(self.chords_history):
                # Check if the current chord is the last one (most recent) in the history
                if idx == len(self.chords_history) - 1 and self.toggle == False:  # Most recent chord
                    Color(0, 0, 0)  # Example: Make it red for visibility. Adjust as per your design preference.
                    line_width = 3  # Thicker line for the most recent chord
                elif self.toggle == True and chord == self.toggled_chord:
                    Color(0,0,0)    # Make the line black if the user scrolls through his shapes.
                    line_width = 3
                else:
                    Color(0.7, 0.7, 0.7)  # Your previous gray color for older chords
                    line_width = 1  # Regular line width for older chords
                for i in range(len(chord)):
                    start_note = chord[i]
                    end_note = chord[(i+1) % len(chord)]
                    start_idx = notes.index(start_note)
                    end_idx = notes.index(end_note)
                    Line(points=[x[start_idx], y[start_idx], x[end_idx], y[end_idx]])

            for i, note in enumerate(notes):
                # Drawing small circles and text labels for notes
                Color(1, 1, 1)
                Ellipse(pos=(x[i]-10, y[i]-10), size=(20, 20))
                Color(0.5, 0.5, 0.5)
                # Add labels here if needed using Kivy's Label widget
                note_label = Label(text=note, center=(round(x[i]), round(y[i])), font_size=12, color=(0, 0, 0, 1))
                self.add_widget(note_label)

class MyApp(MDApp):
    sample_rate = 44100
    chord = ['D#','B','G']
    chord_history = ['C#']
    current_chord_index = 0
    # Load the trained model and label encoder
    model = joblib.load('chord_identifier.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    classifier = Chord_classifier(model, label_encoder)
    
    def build(self):
        self.overlay_mode = False
        # Create the main layout
        self.main_layout = BoxLayout(orientation='vertical', spacing=10, padding=[0, 0, 0, 20])
        self.main_layout.bind(minimum_size=self.main_layout.setter('size'))
        button_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=50, spacing=10, padding=10)
        button_layout.add_widget(Widget())  # Empty widget to take up space
        # Create a header and add it to the main layout
        header = MDBoxLayout(size_hint_y=None, height=50, md_bg_color=(0.106,0.106,0.106))
        # Set background color for the header
        with header.canvas.before:
            Color(0, 0, 0, 0)  # RGBA, for green color in this case
            self.rect = Rectangle(size=header.size, pos=header.pos)
        # Update the Rectangle size and position whenever the header size or position changes
        header.bind(size=self.update_rect, pos=self.update_rect)
        title = Label(text='SoundShapes', color="white")
        header.add_widget(title)

        # Create a ChordCircle widget with a default chord and add it to the layout
        self.chord_circle = ChordCircle(chord=self.chord, overlay_mode=self.overlay_mode, size_hint=(1, 0.3))

        #Chord-name
        self.chord_info = BoxLayout(orientation="horizontal",size_hint_y=None, height=120)
        self.chord_name = Label(text=self.chord_history[0], color="black")

        # Before your toggle_button, let's add the left arrow button
        left_arrow_button = MDIconButton(icon="arrow-left", md_bg_color=(1,1,1),pos_hint={'center_y': 0.5})
        left_arrow_button.bind(on_press=self.scroll_previous_chord)  
        # After your record_button, let's add the right arrow button
        right_arrow_button = MDIconButton(icon="arrow-right", md_bg_color=(1,1,1),pos_hint={'center_y': 0.5})
        right_arrow_button.bind(on_press=self.scroll_next_chord)  

        self.chord_info.add_widget(Widget())
        self.chord_info.add_widget(Widget())
        self.chord_info.add_widget(left_arrow_button)
        self.chord_info.add_widget(self.chord_name)
        self.chord_info.add_widget(right_arrow_button)
        self.chord_info.add_widget(Widget())
        self.chord_info.add_widget(Widget())

        # Add a toggle-button for chord-history
        toggle_button = MDFloatingActionButton(icon="layers", md_bg_color=(0.106,0.167,0.158))
        toggle_button.bind(on_press=self.toggle_overlay_mode)
        button_layout.add_widget(toggle_button)
        # Add a record button
        record_button = MDFloatingActionButton(icon="microphone", md_bg_color=(0.106,0.167,0.158))
        record_button.bind(on_press=self.record_chord)
        button_layout.add_widget(record_button)
        button_layout.add_widget(Widget())  # Empty widget to take up space

        self.recording = False  # Flag to indicate whether recording is in progress
        self.recorded_audio = None  # To store the recorded audio data

        self.main_layout.add_widget(header)
        self.main_layout.add_widget(self.chord_info)
        self.main_layout.add_widget(self.chord_circle)
        self.main_layout.add_widget(button_layout)  

        return self.main_layout

    def scroll_previous_chord(self, instance):
        #update label
        self.current_chord_index -= 1
        self.chord_name.text = self.chord_history[self.current_chord_index]
        notes = self.classifier.get_notes_for_chord(self.chord_history[self.current_chord_index])
        self.chord_circle.highlight_shape(notes)
    
    def scroll_next_chord(self, instance):
        #update label
        self.current_chord_index += 1
        self.chord_name.text = self.chord_history[self.current_chord_index]
        notes = self.classifier.get_notes_for_chord(self.chord_history[self.current_chord_index])
        self.chord_circle.highlight_shape(notes)

    def update_chord_label(self, value):
        self.chord_name.text = value

    def update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def toggle_overlay_mode(self, instance):
        self.overlay_mode = not self.overlay_mode
        self.chord_circle.overlay_enabled = self.overlay_mode  
        if self.overlay_mode:
            instance.icon = "layers-remove"
        else:
            instance.icon = "layers"

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
                #classify the chord
                chord_from_audio = self.classifier.predict_new_chord('recorded_chord.wav',self.model, self.label_encoder)
                #add label to list
                self.chord_history.append(chord_from_audio)
                #derive notes from chord
                notes = self.classifier.get_notes_for_chord(chord_from_audio)
                self.chord = notes
                #draw lines between the notes in the circle
                self.chord_circle.update_chord(self.chord)
                #update label
                self.current_chord_index += 1
                self.update_chord_label(chord_from_audio)


if __name__ == '__main__':
    MyApp().run()