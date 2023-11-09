import os
import librosa
import numpy as np
import scipy.signal
from scipy.signal import butter, lfilter, sosfilt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(y, sr, cutoff=100, order=2):
    nyquist = 0.5 * sr
    norm_cutoff = cutoff / nyquist
    sos = scipy.signal.butter(order, norm_cutoff, btype='highpass', analog=False, output='sos')
    return scipy.signal.sosfilt(sos, y)


# Define a function to extract features from an audio file
def extract_features(audio_file):
    # Load the audio file
    chord, fs = librosa.load(audio_file, sr=None)
    
    # Decompose the audio signal into harmonic and percussive components
    harmonic, percussive = librosa.effects.hpss(chord)
    
    # Compute the constant-Q transform (CQT)
    # Here, we assume that fmin is C1, which is a common choice. You may change this as needed.
    C = librosa.cqt(y=harmonic, sr=fs, fmin=librosa.note_to_hz('C1'))
    
    # Convert the complex CQT output into magnitude, which represents the energy at each CQT bin
    # Summing across the time axis gives us the aggregate energy for each pitch bin
    pitch_sum = np.abs(C).sum(axis=1)
    
    return pitch_sum


# Define the paths to your "train" and "test" directories
train_data_dir = "Training"
test_data_dir = "Test"

# Initialize empty lists to store features and labels
features = []
labels = []

# Iterate through the training data
for root, dirs, files in os.walk(train_data_dir):
    for file in files:
        if file.endswith(".wav"):
            # Extract features from the audio file
            feature_vector = extract_features(os.path.join(root, file))
            # Append the feature vector to the features list
            features.append(feature_vector)
            # Extract the chord label from the subfolder name
            label = os.path.basename(os.path.dirname(os.path.join(root, file)))
            labels.append(label)

# Encode chord labels to numerical values
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, encoded_labels, test_size=0.2, random_state=42)

# Train a machine learning model (Random Forest in this example)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Decode the predicted labels
predicted_labels = label_encoder.inverse_transform(y_pred)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Print actual chord labels and their corresponding notes
actual_chords = label_encoder.inverse_transform(y_test)
for actual_chord, predicted_chord in zip(actual_chords, predicted_labels):
    print(f"Actual Chord: {actual_chord}, Predicted Chord: {predicted_chord}")

#dumps the model into a file.
joblib.dump(model, 'chord_identifier.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

