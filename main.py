import librosa
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def get_notes_for_chord(chord):
    
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

def draw_chr_circle(chord):
    # Define the notes and their positions
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    num_notes = len(notes)
    theta = np.linspace(0, 2*np.pi, num_notes, endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal', 'box')

    # Plot the chromatic circle
    ax.plot(x, y, color='gray')  # Circle
    for i, note in enumerate(notes):
        ax.text(x[i]*1.1, y[i]*1.1, note, ha='center', va='center')

    for i in range(num_notes):
        start_idx = i
        end_idx = (i+1) % num_notes  # This ensures that after the last note, we loop back to the first note
        ax.plot([x[start_idx], x[end_idx]], [y[start_idx], y[end_idx]], color='gray')

    for i in range(len(chord)):
        start_note = chord[i]
        end_note = chord[(i+1) % len(chord)]  # wrap around to create the last segment
        start_idx = notes.index(start_note)
        end_idx = notes.index(end_note)
        ax.plot([x[start_idx], x[end_idx]], [y[start_idx], y[end_idx]], 'r-')

    # Some additional formatting
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')
    plt.show()

def extract_features(audio_file):
    chord, fs = librosa.load(audio_file, sr=None)
    chord_emphasized = librosa.effects.preemphasis(chord, coef=0.97)
    chromagram = librosa.feature.chroma_cens(y=chord_emphasized, sr=fs, hop_length=512)
    pitch_sum = chromagram.sum(axis=1)
    return pitch_sum

# Define a function to predict the chord for a new audio file
def predict_new_chord(audio_file_path, model, label_encoder):
    # Extract features from the new audio file
    feature_vector = extract_features(audio_file_path)

    # Reshape the feature vector to match the model's input shape
    feature_vector = feature_vector.reshape(1, -1)

    # Use the trained model to make a prediction
    predicted_label = model.predict(feature_vector)

    # Decode the predicted label to get the chord name
    predicted_chord = label_encoder.inverse_transform(predicted_label)

    return predicted_chord[0]

def main():
    
    # Load the trained model and label encoder
    model = joblib.load('chord_identifier.pkl')
    label_encoder = joblib.load('label_encoder.pkl')

    # Example: Predict the chord for a new audio file
    new_audio_file_path = 'Test\Dm\Dm_AcousticGuitar_RodrigoMercador_2.wav'
    
    predicted_chord = predict_new_chord(new_audio_file_path, model, label_encoder)
    print(f"Predicted Chord for the new audio file: {predicted_chord}")
    draw_chr_circle(get_notes_for_chord(predicted_chord))

if __name__ == "__main__":
    main()
