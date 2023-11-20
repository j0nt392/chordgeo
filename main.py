import pygame

from kivy.app import App
from kivymd.app import MDApp
from kivymd.uix.button import MDRoundFlatButton, MDFloatingActionButton
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.floatlayout import FloatLayout
from kivymd.uix.button import MDIconButton
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.graphics import Color, Line, Ellipse, Rectangle
from kivy.core.window import Window
from kivy.core.text import LabelBase

# Registering the custom font (make sure the font file is in your app directory)
LabelBase.register(name='Roboto', 
                   fn_regular='Lato\Lato-LightItalic.ttf')

from plyer import filechooser 
import pyaudio
import struct
from scipy.fftpack import fft
import time 
import threading

import librosa
import numpy as np
import joblib

from kivy.clock import Clock
import sounddevice as sd
import numpy as np
import wave
import warnings

import matplotlib as plt
from sklearn.tree import plot_tree

import os 
os.environ["KIVY_NO_CONSOLELOG"] = "1"

# # Set the window size
# Window.size = (360, 640)  # width x height

# # Optional: Disable window resizing
# Window.resizable = True

class AudioPlayer():
    def __init__(self, audio_path):
        self.audio_path = audio_path
        pygame.mixer.init()
        self.track = pygame.mixer.Sound(audio_path)
    
    def play(self):
        self.track.play()

    def stop(self):
        self.track.stop()

    def fast_forward(self):
        pass

    def rewind(self):
        pass

class WaveformWidget(Widget):
    def __init__(self, audio_path, **kwargs):
        super(WaveformWidget, self).__init__(**kwargs)
        self.audio_path = audio_path
        self.size_hint = (0.8, 0.2)  # Occupy 80% width and 20% height of the parent
        self.pos_hint = {'center_x': 0.5, 'center_y': 0.5}  #
        self.waveform_points = self.waveform_coordinates()
        self.draw_waveform()

    def draw_waveform(self):
        with self.canvas:
            Color(0.2,0.2,0.2)
            Line(points=self.waveform_points)
            
    def waveform_coordinates(self):
        y, sr = librosa.load(self.audio_path)
        waveform = librosa.feature.melspectrogram(y=y, sr=sr)
        waveform = np.mean(waveform, axis=0)

        max_height = self.height
        points = []
        waveform_width = self.width
        waveform_height = self.height
        horizontal_offset = (Window.width - waveform_width) / 2
        vertical_offset = max_height 
        points = []
        for i, val in enumerate(waveform):
            # Apply horizontal offset to x
            x = i * (waveform_width / len(waveform)) + horizontal_offset
            # Center y within the widget
            y = (val / np.max(waveform)) * max_height / 2 + vertical_offset + 20
            points.extend([x, y])
            
        return points

class AudioStream(object):
    def __init__(self, chord_circle, label, history):
        self.history = history
        self.label = label
        self.chord = None
        self.chord_circle = chord_circle
        # stream constants
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100
        self.streaming = False
        self.classifier = Chord_classifier()  # Assuming Chord_classifier is defined elsewhere
        self.p = pyaudio.PyAudio()
        self.notes = []
        self.audio_buffer = []
        self.buffer_length = 44100 * 2  # Buffer length in samples
        self.buffer_lock = threading.Lock()  # Lock for thread-safe buffer access
        self.stream = self.p.open(format=self.FORMAT,
                            channels=self.CHANNELS,
                            rate=self.RATE,
                            input=True,
                            output=True,
                            frames_per_buffer=self.CHUNK)

    def stream_audio(self):
        print('stream started')

        while self.streaming:
            data = self.stream.read(self.CHUNK)
            data_int = struct.unpack(str(self.CHUNK) + 'f', data)
            
            with self.buffer_lock:  # Lock for thread-safe buffer access
                self.audio_buffer.extend(data_int)

    def analyze_buffer(self):
        while self.streaming:
            if len(self.audio_buffer) >= self.buffer_length:
                with self.buffer_lock:  # Lock for thread-safe buffer access
                    audio_signal = np.array(self.audio_buffer[:self.buffer_length], dtype=np.float32)
                    self.audio_buffer = self.audio_buffer[self.buffer_length:]

                self.update_label(audio_signal)

    def update_label(self, pitch_sum):
        self.chord = self.classifier.predict_new_chord(pitch_sum, self.RATE)
        self.label.text = self.chord
        self.notes = self.classifier.get_notes_for_chord(self.chord)
        Clock.schedule_once(self.update_circle_with_notes, 0)

    def update_circle_with_notes(self, dt):
        # Call the update_circle method of the ChordCircle widget
        self.chord_circle.update_chord(self.notes)

    def toggle_stream(self):
        if self.streaming:
            print('stream closed')
            self.streaming = False
            if self.audio_thread.is_alive():
                self.audio_thread.join()
            if self.analysis_thread.is_alive():
                self.analysis_thread.join()
        else:
            self.streaming = True
            self.audio_thread = threading.Thread(target=self.stream_audio)
            self.analysis_thread = threading.Thread(target=self.analyze_buffer)
            self.audio_thread.start()
            self.analysis_thread.start()
   
class Chord_preprocessing():
    def __init__(self):
        pass

    def adjust_loudness(self, audio, fs, target_rms=-15):
        # Calculate the RMS energy of the audio
        rms = np.sqrt(np.mean(audio**2))

        # Calculate the gain needed to reach the target RMS level
        gain = 10**((target_rms - 20 * np.log10(rms)) / 20)

        # Apply the gain to the audio
        adjusted_audio = audio * gain

        # Specify the output filename
        output_filename = "adjusted_audio.wav"

        # Export the adjusted audio to a WAV file
        #sf.write(output_filename, adjusted_audio, fs)

        return adjusted_audio

class Chord_classifier():
    '''Uses the chord_identifier.pkl model to classify any chord.'''
    def __init__(self):
        self.model = joblib.load('chord_identifier.pkl')
        self.label_encoder = joblib.load('label_encoder.pkl')
        self.preprocessing = Chord_preprocessing()
    
    def get_notes_for_chord(self, chord):
        '''takes a chord (C#) and gives you the triad notes in that chord'''
        chord_notes_mapping = {
        'Ab': ['Ab', 'Eb', 'C'],
        'A': ['A', 'Db', 'E'],
        'Am': ['A', 'C', 'E'],
        'B': ['B', 'Gb', 'Eb'],
        'Bb': ['Bb', 'D', 'F'],
        'Bdim': ['B', 'D', 'F'],
        'C': ['C', 'E', 'G'],
        'Db': ['Db', 'Ab', 'F'],
        'D': ['D', 'A', 'Gb'],
        'Dm': ['D', 'F', 'A'],
        'Eb': ['Eb', 'Bb', 'G'],
        'E': ['E', 'B', 'Ab'],
        'Em': ['E', 'G', 'B'],
        'F': ['F', 'A', 'C'],
        'Gb': ['Gb', 'Db', 'Bb'],
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

    def _extract_features(self, audio_file, fs):
        audio = None
        if type(audio_file) == str:
            audio, fs = librosa.load(audio_file, sr = None)
        else:
            audio = audio_file
        #preprocessing
        harmonic, percussive = librosa.effects.hpss(audio)
        
        # Compute the constant-Q transform (CQT)
        C = librosa.cqt(y=harmonic, sr=fs, fmin=librosa.note_to_hz('C1'), hop_length=256, n_bins=84)
        
        # Convert the complex CQT output into magnitude, which represents the energy at each CQT bin
        # Summing across the time axis gives us the aggregate energy for each pitch bin
        pitch_sum = np.abs(C).sum(axis=1)
        
        return pitch_sum

    def predict_new_chord(self, audio_file_path, fs):
        # Extract features from the new audio file
        feature_vector = self._extract_features(audio_file_path, fs)
        # # Reshape the feature vector to match the model's input shape
        feature_vector = feature_vector.reshape(1, -1)
        try:
            predicted_label = self.model.predict(feature_vector)
            predicted_chord = self.label_encoder.inverse_transform(predicted_label)
            return predicted_chord[0]       
        except Exception as e:
            return "Error during prediction: %s", str(e)

    def analyze_chord_progression(self, audio_file, buffer_length=1, hop_length=0.2):
        # Load the audio file
        y, sr = librosa.load(audio_file, sr=None)
        # Calculate the number of samples per buffer
        buffer_samples = int(buffer_length * sr)
        hop_samples = int(hop_length * sr)
        chords = []
        # Start at the beginning and hop through the file
        for start in range(0, len(y), hop_samples):
            end = start + buffer_samples
            # Make sure we don't go past the end of the audio file
            if end <= len(y):
                buffer = y[start:end]
                # Predict the chord for this buffer
                chord = self.predict_new_chord(buffer, sr)
                chords.append(chord)
            else:
                break  # We've reached the end of the audio
        # Return the list of chords
        return chords

class ChordCircle(Widget):
    #center_x = Window.width * 0.23 #iphone
    #center_y = Window.height - 300 #iphone
    center_x = Window.width/2
    center_y = Window.height/2
    radius = min(Window.width, Window.height) * 0.25 - 20  # Deduct 20 to account for the small circles

    def __init__(self, chord, circle_type, overlay_mode=False, **kwargs):
        super(ChordCircle, self).__init__(**kwargs)
        self.circle_type = circle_type
        self.chords_history = [chord]
        self.overlay_enabled = overlay_mode
        self.toggle = False
        #self.toggled_chord_index = None
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
        #self.canvas.clear()
        self.draw_chr_circle()

    def draw_chr_circle(self):
        self.canvas.clear()

        color_scheme = {
            'C, E, G': (1, 0, 0),          # Red
            'Db, Ab, F': (1, 0.5, 0),  # Orange-Red
            'D, A, Gb': (1, 1, 0),          # Yellow
            'Eb, Bb, G': (0.5, 1, 0),       # Yellow-Green
            'E, B, Ab': (0, 1, 0),          # Green
            'F, A, C': (0, 1, 1),          # Cyan
            'Gb, Db, Bb': (0, 0.5, 1),       # Light Blue
            'G, B, D': (0, 0, 1),          # Blue
            'Ab, Eb, C': (0.5, 0, 1),       # Purple
            'A, Db, E': (1, 0, 1),          # Magenta
            'Bb, D, F': (1, 0, 0.5),       # Pink
            'B, Gb, Eb': (1, 0.5, 0.5),      # Salmon
            'D, F, A': (1, 0.5, 0),
            'A, C, E': (1, 1, 0),
            'B, D, F': (0, 1, 1),
            'E, G, B': (1, 0, 0.5)
        }
        
        notes = []
        if self.circle_type == 'Chromatic circle':
            notes = ['Eb', 'D', 'Db', 'C', 'B', 'Bb','A', 'Ab', 'G', 'Gb', 'F', 'E']

        elif self.circle_type == 'Circle of fifths':
            notes = ['Eb', 'Bb', 'F', 'C', 'G', 'D', 'A', 'E', 'B', 'Gb', 'Db', 'Ab']

        num_notes = len(notes)
        theta = np.linspace(0, 2*np.pi, num_notes, endpoint=False)
        x = np.cos(theta) * self.radius + self.center_x
        y = np.sin(theta) * self.radius + self.center_y
        highlighted_chord_idx = None

        with self.canvas:
            Color(0.5, 0.5, 0.5)
            line_width = 1.05  # Thicker line for the most recent chord
            print(self.chords_history)
            Line(circle=(self.center_x, self.center_y, self.radius), width=line_width)
            #for i, note in enumerate(notes):
                #Line(points=[x[i], y[i], x[(i+1) % num_notes], y[(i+1) % num_notes]])

            for idx, chord in enumerate(self.chords_history):
                stringified = ', '.join(chord)
                # Check if the current chord is the last one (most recent) in the history
                if idx == len(self.chords_history) - 1 and self.toggle == False:  # Most recent chord
                    if len(chord) > 0:
                        # for colorwheel Color(*color_scheme[stringified])
                        Color(0.5, 0.5, 0.5)

                # Self.Toggle checks if you have scrolled 
                elif self.toggle == True and chord == self.toggled_chord:
                    highlighted_chord_idx = idx
                    continue
                    
                else:
                    if len(chord) > 0:
                        #Color(*color_scheme[stringified])
                        Color(0.5,0.5,0.5)
                for i in range(len(chord)):
                    start_note = chord[i]
                    end_note = chord[(i+1) % len(chord)]
                    start_idx = notes.index(start_note)
                    end_idx = notes.index(end_note)
                    Line(points=[x[start_idx], y[start_idx], x[end_idx], y[end_idx]], width=line_width)
            
            # Highlight the chord scrolled to
            if highlighted_chord_idx is not None:
                chord = self.chords_history[highlighted_chord_idx]
                Color(0,0,0)
                for i in range(len(chord)):
                    start_note = chord[i]
                    end_note = chord[(i+1) % len(chord)]
                    start_idx = notes.index(start_note)
                    end_idx = notes.index(end_note)
                    Line(points=[x[start_idx], y[start_idx], x[end_idx], y[end_idx]])
                self.toggle = False
                highlighted_chord_idx = None

            for i, note in enumerate(notes):
                # Drawing small circles and text labels for notes
                Color(1, 1, 1)
                Ellipse(pos=(x[i]-10, y[i]-10), size=(20, 20))
                Color(0.5, 0.5, 0.5)
                # Add labels here if needed using Kivy's Label widget
                note_label = Label(text=note, center=(round(x[i]), round(y[i])), font_size=15, color=(0, 0, 0, 1))
                self.add_widget(note_label)

class MyApp(MDApp):
    def build(self):
        self.chord = []
        self.chord_history = []
        self.current_chord_index = 0
        self.recording = False
        self.overlay_mode = False
        self.recorded_audio = None  # To store the recorded audio data
        self.dropdown_visible = False
        #Window.clearcolor = (0.196, 0.196, 0.196, 1)
        # Create the main layout
        self.main_layout = FloatLayout()

        '''header and dropdown configuration'''
        self.header = MDBoxLayout(size_hint=(1, None), height=50, pos_hint={'top':1}, md_bg_color=(0.2,0.2,0.2))
        header_label = Label(text='SoundShapes', size_hint_x=1, size=(Window.width, self.header.height), halign='left', valign='middle', padding=15)
        header_label.bind(size=lambda *args: setattr(header_label, 'text_size', header_label.size))

        self.menu_button = MDIconButton(icon='menu', size_hint=(None,None), pos_hint={'center_y':0.5}, theme_text_color='Custom',text_color=[1,1,1,1])
        self.menu_button.bind(on_press=self.toggle_dropdown)
        
        dropdown_top = 1 - (self.header.height / Window.height)
        self.dropdown = BoxLayout(size_hint=(1, None), height=0, pos_hint={'top': dropdown_top})
        self.dropdown.height = 0  # Initially, the dropdown is not visible.

        '''Circle type description'''
        self.circle_type_label = Label(text='Chromatic circle', font_size="28sp", font_name="Roboto", color="black", size_hint=(None,None), size=(Window.width * 0.3, 80),pos_hint={'x': 0.35, 'center_y': 0.8})
        
        '''Chord circle'''
        self.chord_circle = ChordCircle(chord=self.chord, overlay_mode=self.overlay_mode, circle_type='chromatic_circle', size_hint=(1, 0.3))
        
        '''Chord-name label and arrows to scroll chords.'''
        #self.chord_info = BoxLayout(orientation="horizontal",size_hint_y=None, height=120)
        self.chord_name = Label(text='Chords', color="black", size_hint=(None,None), size=(Window.width * 0.3, 50),pos_hint={'x': 0.35, 'center_y': 0.2})
        self.left_arrow_button = MDIconButton(icon="arrow-left", md_bg_color=(1,1,1),pos_hint={'x': 0.375, 'center_y': 0.2})
        self.left_arrow_button.bind(on_press=self.scroll_previous_chord)  
        self.right_arrow_button = MDIconButton(icon="arrow-right", md_bg_color=(1,1,1),pos_hint={'x': 0.555, 'center_y': 0.2})
        self.right_arrow_button.bind(on_press=self.scroll_next_chord)  

        '''Here is the layout for the buttons at the bottom.'''
        self.button_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=80, spacing=10, padding=20)
        self.button_layout.add_widget(Widget())  # Empty widget to take up space

        '''Toggle button for seeing overlapping shapes'''
        toggle_button = MDFloatingActionButton(icon="layers",md_bg_color=(0.2, 0.2, 0.2))
        toggle_button.bind(on_press=self.toggle_overlay_mode)
        self.button_layout.add_widget(toggle_button)

        '''Clears all shapes'''
        clear_button = MDFloatingActionButton(icon="delete", md_bg_color=(0.2,0.2,0.2))
        clear_button.bind(on_press=self.clear_patterns)
        self.button_layout.add_widget(clear_button)
        
        '''Record button'''
        self.record_button = MDFloatingActionButton(icon="microphone", size_hint=(0.4,2), type='small', md_bg_color=(0.2,0.2,0.2))
        self.record_button.bind(on_press=self.record_audio)
        self.button_layout.add_widget(self.record_button)

        '''Add folder for loading song'''
        folder_button = MDFloatingActionButton(icon='folder', size_hint=(None, None), size=("29dp", "20dp"), md_bg_color=(0.2,0.2,0.2))
        self.button_layout.add_widget(folder_button)
        folder_button.bind(on_press= self.load_song)

        input_button = MDFloatingActionButton(icon='keyboard', md_bg_color=(0.2,0.2,0.2))
        self.button_layout.add_widget(input_button)
        input_button.bind(on_press= self.input_chords)
        self.button_layout.add_widget(Widget()) 

        '''Add everything to main layout'''
        self.header.add_widget(header_label)
        self.header.add_widget(self.menu_button)
        self.main_layout.add_widget(self.header)
        self.main_layout.add_widget(self.circle_type_label)
        
        self.main_layout.add_widget(self.chord_circle)
        
        self.main_layout.add_widget(self.left_arrow_button)
        self.main_layout.add_widget(self.chord_name)
        self.main_layout.add_widget(self.right_arrow_button)
        self.main_layout.add_widget(self.dropdown)
        #self.main_layout.add_widget(self.chord_info)
        self.main_layout.add_widget(self.button_layout)  
        
        # Create an audiorecorder that streams audio
        self.recorder = AudioStream(chord_circle=self.chord_circle, label=self.chord_name, history=self.chord_history)
        return self.main_layout

    def toggle_dropdown(self, instance):
        options = {
            'Chromatic circle': 'Chromatic circle',
            'Circle of fifths': 'Circle of fifths',
            'Coltrane circle': 'coltrane_circle',
            'Settings' : 'settings'
        }

        if self.dropdown_visible:
            self.dropdown.height = 0
            self.dropdown.clear_widgets()
        else:
            self.dropdown.orientation = 'vertical'
            for i, value in options.items():
                btn = Button(text=f'{i}', height=50, size_hint_y=None,
                            background_normal='', background_down='', background_color=(0.106,0.106,0.106))
                
                btn.bind(on_press = lambda inst, c=value: self.circletype_callback(inst,c))
                self.dropdown.add_widget(btn)
            
            self.dropdown.height = sum(btn.height for btn in self.dropdown.children)
        
        self.dropdown_visible = not self.dropdown_visible

    def circletype_callback(self, instance, c):
        self.chord_circle.circle_type = c
        self.circle_type_label.text = c
        self.chord_circle.draw_chr_circle()

    def scroll_previous_chord(self, instance):
        #update label
        self.current_chord_index -= 1
        self.chord_name.text = self.chord_history[self.current_chord_index]
        notes = self.recorder.classifier.get_notes_for_chord(self.chord_history[self.current_chord_index])
        self.chord_circle.highlight_shape(notes)
    
    def scroll_next_chord(self, instance):
        #update label
        self.current_chord_index += 1
        self.chord_name.text = self.chord_history[self.current_chord_index]
        notes = self.recorder.classifier.get_notes_for_chord(self.chord_history[self.current_chord_index])
        self.chord_circle.highlight_shape(notes)

    def update_chord_label(self, value):
        self.chord_name.text = value

    def update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def clear_patterns(self, insance):
        self.chord_history = []
        self.chord_circle.chords_history = []
        self.chord_name.text = " "
        self.chord_circle.draw_chr_circle()

    def toggle_overlay_mode(self, instance):
        self.overlay_mode = not self.overlay_mode
        self.chord_circle.overlay_enabled = self.overlay_mode  
        if self.overlay_mode:
            instance.icon = "layers-remove"
        else:
            # Clear patterns 
            instance.icon = "layers"

    def record_audio(self, instance):
        if self.recording == False and self.recorder.streaming == False:
            instance.md_bg_color = (0.9, 0.3, 0.3, 1)
            instance.icon = 'stop'
            self.recorder.toggle_stream()
        elif self.recorder.streaming == True:
            instance.md_bg_color=(0.2,0.2,0.2)
            instance.icon = 'microphone'
            self.recorder.toggle_stream()

    def load_song(self, instance):
        filechooser.open_file(on_selection=self.handle_selection)

    def handle_selection(self, selection):
        if selection:
            if self.overlay_mode == False:
                self.chord_circle.chords_history = []
                self.chord_history = []
            self.chord_circle.overlay_enabled = True
            filepath = selection[0]
            chord_progression = self.recorder.classifier.analyze_chord_progression(filepath)
            for chord in chord_progression:
                notes = self.recorder.classifier.get_notes_for_chord(chord)
                self.chord = notes
                self.chord_name.text = chord_progression[-1]
                self.chord_circle.update_chord(self.chord)
            self.chord_circle.overlay_enabled = False

            waveform_widget = WaveformWidget(filepath)
            self.main_layout.add_widget(waveform_widget)
            self.right_arrow_button.opacity = 0
            self.left_arrow_button.opacity = 0
            self.chord_name.opacity = 0

            audio_player = AudioPlayer(filepath)
            audio_player.play()


    def input_chords(self, instance):
        self.chord_circle.overlay_enabled = True
        self.chord_circle.update_chord(['A', 'Db', 'F'])  # A Augmented
        self.chord_circle.update_chord(['Bb', 'D', 'Gb'])  # Bb Augmented
        self.chord_circle.update_chord(['B', 'Eb', 'G'])  # B Augmented
        self.chord_circle.update_chord(['C', 'E', 'Ab'])  # C Augmented
        self.chord_circle.update_chord(['Db', 'F', 'A'])  # Db Augmented
        self.chord_circle.update_chord(['D', 'Gb', 'Bb'])  # D Augmented
        self.chord_circle.update_chord(['Eb', 'G', 'B'])  # Eb Augmented
        self.chord_circle.update_chord(['E', 'Ab', 'C'])  # E Augmented
        self.chord_circle.update_chord(['F', 'A', 'Db'])  # F Augmented
        self.chord_circle.update_chord(['Gb', 'Bb', 'D'])  # Gb Augmented
        self.chord_circle.update_chord(['G', 'B', 'Eb'])  # G Augmented
        self.chord_circle.update_chord(['Ab', 'C', 'E'])  # Ab Augmented

if __name__ == '__main__':
    app = MyApp().run()
