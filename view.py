import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
