import numpy as np 
import librosa 
from view import draw_chr_circle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import re
import math

def draw_chr_circle(chord):
    # Define the notes and their positions
    notes = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']

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

def get_hz():
    chord, fs = librosa.load('Training\Am\Am_acousticguitar_Mari_1.wav', sr=None)

    # Compute chroma features
    chroma = librosa.feature.chroma_stft(y=chord, sr=fs)

    # Compute chromagram (if needed)
    chromagram = librosa.feature.chroma_cens(y=chord, sr=fs)

    # Sum the chromagram to get the pitch content
    pitch_sum = chromagram.sum(axis=1)

    # Define the mapping of pitch classes to note names
    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # Find the pitches with the highest energy
    top_pitches = []
    for _ in range(3):  # You want to find the top 3 pitches
        max_pitch = pitch_classes[pitch_sum.argmax()]
        top_pitches.append(max_pitch)
        pitch_sum[pitch_sum.argmax()] = 0  # Set the maximum value to 0 to find the next maximum

    print(f"Pitches in the audio: {', '.join(top_pitches)}")


def map_frequencies(frequencies:list):
    notes = {
        "C2": 65.41,
        "C#2": 69.30,
        "D2": 73.42,
        "D#2": 77.78,
        "E2": 82.41,
        "F2": 87.31,
        "F#2": 92.50,
        "G2": 98.00,
        "G#2": 103.83,
        "A2": 110.00,
        "A#2": 116.54,
        "B2": 123.47,

        "C3": 130.81,
        "C#3": 138.59,
        "D3": 146.83,
        "D#3": 155.56,
        "E3": 164.81,
        "F3": 174.61,
        "F#3": 185.00,
        "G3": 196.00,
        "G#3": 207.65,
        "A3": 220.00,
        "A#3": 233.08,
        "B3": 246.94,

        "C4": 261.63,
        "C#4": 277.18,
        "D4": 293.66,
        "D#4": 311.13,
        "E4": 329.63,
        "F4": 349.23,
        "F#4": 369.99,
        "G4": 392.00,
        "G#4": 415.30,
        "A4": 440.00,
        "A#4": 466.16,
        "B4": 493.88,
        }
    chord_notes = []
    
    #remove close frequencies
    for i, f in enumerate(notes):
        for j in frequencies:
            if abs(notes[f] - j) <= 1:
                chord_notes.append(f)
    
    pattern = r'[0-9]'

    #remove digits 
    for i, note in enumerate(chord_notes):
        chord_notes[i] = re.sub(pattern, '', note)
    # Match all digits in the string and replace them with an empty string
    return chord_notes

def main():
    get_hz()

if __name__ == ("__main__"):
    main()