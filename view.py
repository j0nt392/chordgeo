from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.graphics import Color, Line, Ellipse
from kivy.core.window import Window
from kivy.animation import Animation
import numpy as np

# Set the window size
Window.size = (360, 640)  # width x height

# Optional: Disable window resizing
Window.resizable = True

class ChordCircle(Widget):
    def __init__(self, chord, **kwargs):
        super(ChordCircle, self).__init__(**kwargs)
        self.chord = chord
        self.draw_chr_circle()

    def draw_chr_circle(self):
        # Define the notes and their positions
        center_x = round(Window.width * 0.23)
        center_y = round(Window.height - 300)
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        num_notes = len(notes)
        theta = np.linspace(0, 2*np.pi, num_notes, endpoint=False)
        radius = min(Window.width, Window.height) * 0.25 - 20  # Deduct 20 to account for the small circles
        x = np.cos(theta) * radius + center_x
        y = np.sin(theta) * radius + center_y

        with self.canvas:
            Color(0.5, 0.5, 0.5)
            Line(circle=(center_x, center_y, radius))
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
    def build(self):
        return ChordCircle(chord=['D#', 'G', 'B'])  # Example chord

if __name__ == '__main__':
    MyApp().run()


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