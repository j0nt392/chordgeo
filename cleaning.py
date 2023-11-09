from pydub import AudioSegment
from scipy.io import wavfile
from scipy.signal import resample
import numpy as np
import os

class DataCleaning():
    def __init__(self, root_directory):
        self.root_directory = root_directory
        self.edit_files('pitchup')
    # Define a function to remove silence from an audio file
    def _remove_silence(self, input_file, silence_threshold=-40):
        audio = AudioSegment.from_file(input_file, format="wav")
        non_silent = audio - silence_threshold
        non_silent = non_silent.set_frame_rate(audio.frame_rate)
        
        # Overwrite the original file with the processed audio
        non_silent.export(input_file, format="wav")
    
    def pitch_shift(self, input_file, output_file):
        # Load the audio file
        sample_rate, audio_data = wavfile.read(input_file)

        # Define the pitch shift factor (2.0 for an octave shift up)
        pitch_shift_factors = [0.5, 2.0]
        
        for x in pitch_shift_factors:
            output_file_name = output_file + str(x) + '.wav'
            # Calculate the new length of the audio data after pitch shift
            new_length = int(len(audio_data) / x)
            # Perform pitch shift using resample
            pitch_shifted_audio = resample(audio_data, new_length)
            # Save the pitch-shifted audio to the output file
            wavfile.write(output_file_name, sample_rate, pitch_shifted_audio.astype(np.int16))

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
        sf.write(output_filename, adjusted_audio, fs)

        return adjusted_audio

    # Define a function to pitch shift an audio file up an octave
    def time_stretch(self, input_file, output_file):
        audio = AudioSegment.from_file(input_file, format="wav")
        shifted_audio = audio.speedup(playback_speed=2.0)
        shifted_audio.export(output_file, format="wav")
    
    def edit_files(self, cleaning):
        # Loop through the subfolders
        #for folder_name in os.listdir(self.root_directory):
            #folder_path = os.path.join(self.root_directory, folder_name)
            
            # Check if the item in the root directory is a directory itself
            #if os.path.isdir(folder_name):
                # Loop through the files in the subfolder
        for filename in os.listdir(self.root_directory):
            if filename.endswith(".wav"):  # Adjust the file extension as needed
                input_file = os.path.join(self.root_directory, filename)
                
                if cleaning == 'pitchup':
                    output_file = os.path.join(self.root_directory, f"_pitchshifted{filename}")
                    self.pitch_shift(input_file, output_file)
                    print(f"pitchedup: {input_file}")
                elif cleaning == 'removesilence':
                    self._remove_silence(input_file)
                    print(f"Processed and overwrote: {input_file}")
        if cleaning == 'pitchup':
            print("pitchup completed.")

        elif cleaning == 'removesilence':
            print("Silence removal completed.")


#cleaner = DataCleaning('Training\Gb')
