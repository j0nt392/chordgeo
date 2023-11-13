import pyaudio
import struct
import librosa
import numpy as np
import kivy
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
import threading
import time
import joblib 

class AudioStream(object):
    def __init__(self, label):
        self.label = label
        # stream constants
        self.CHUNK = 1024 * 2
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.pause = False
        self.model = joblib.load("chord_identifier.pkl")
        self.encoder = joblib.load("label_encoder.pkl")

        # stream object
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            output=True,
            frames_per_buffer=self.CHUNK,
        )
        self.audio_thread = threading.Thread(target=self.start_stream)
        self.audio_thread.daemon = True  # Allow the thread to be terminated when the program exits
        self.audio_thread.start()

    def start_stream(self):
        print('stream started')
        frame_count = 0
        start_time = time.time()
        while True:
            if self.pause == False:
                data = self.stream.read(self.CHUNK)
                data_int = struct.unpack(str(2 * self.CHUNK) + 'B', data)

                # Calculate chromagram and pitch sum
                audio_signal = np.array(data_int, dtype=np.float32) / 32768.0  # Normalize to floating-point
                # Decompose the audio signal into harmonic and percussive components
                # harmonic, percussive = librosa.effects.hpss(audio_signal)

                # # Compute the constant-Q transform (CQT)
                # # Here, we assume that fmin is C1, which is a common choice. You may change this as needed.
                # C = librosa.cqt(y=harmonic, sr=self.RATE, fmin=librosa.note_to_hz('C1'), hop_length=512, n_bins=81)

                # # Convert the complex CQT output into magnitude, which represents the energy at each CQT bin
                # # Summing across the time axis gives us the aggregate energy for each pitch bin

                # pitch_sum = np.abs(C).sum(axis=1)
                # pitch_sum = pitch_sum.reshape(1, -1)
                
                # Update the label text in the Kivy UI
                Clock.schedule_once(lambda dt: self.update_label(audio_signal), 0)

                frame_count += 1

            else:
                print("stream paused")

    def update_label(self, audio_signal):
        #self.label.text = f'Pitch Sum: {pitch_sum[0]}'
        self.predict_new_chord(audio_signal, self.RATE)

    def toggle_pause(self):
        # Toggle the 'pause' variable
        self.pause = not self.pause

    def extract_features(self, audio_file, fs):
        #preprocessing
        #processed_chord = self.preprocessing.adjust_loudness(chord, fs)

        # Decompose the audio signal into harmonic and percussive components
        harmonic, percussive = librosa.effects.hpss(audio_file)
        
        vqt = librosa.vqt(harmonic, sr=fs)

        # Compute the constant-Q transform (CQT)
        # Here, we assume that fmin is C1, which is a common choice. You may change this as needed.
        chroma = librosa.feature.chroma_cqt(C=vqt)

        # Convert the complex CQT output into magnitude, which represents the energy at each CQT bin
        # Summing across the time axis gives us the aggregate energy for each pitch bin
        pitch_sum = np.abs(chroma).sum(axis=1)

        # Define the desired dimensionality (e.g., 84)
        desired_dimension = 84

        # Zero-pad the pitch_sum to the desired dimensionality
        padded_pitch_sum = np.pad(pitch_sum, (0, desired_dimension - len(pitch_sum)), mode='constant')

        return padded_pitch_sum
        
        #return pitch_sum
    
    def predict_new_chord(self, audio_file_path, fs):
        # Extract features from the new audio file
        feature_vector = self.extract_features(audio_file_path, fs)

        # Reshape the feature vector to match the model's input shape
        feature_vector = feature_vector.reshape(1, -1)

        # Use the trained model to make a prediction
        predicted_label = self.model.predict(feature_vector)

        # Decode the predicted label to get the chord name
        predicted_chord = self.encoder.inverse_transform(predicted_label)

        print(predicted_chord[0])

class AudioPitchApp(App):
    def build(self):
        layout = BoxLayout(orientation='vertical')
        self.label = Label(text="Pitch Sum: 0")
        layout.add_widget(self.label)

        # Create an instance of AudioStream and pass the label
        self.audio_stream = AudioStream(self.label)

        return layout

if __name__ == '__main__':
    AudioPitchApp().run()