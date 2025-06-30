import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import tempfile
from opensoundscape.audio import Audio
from opensoundscape.spectrogram import Spectrogram
from jupyter_bbox_widget import BBoxWidget
import ipywidgets as widgets
from IPython.display import display, clear_output

class AudioBBoxAnnotator:
    def __init__(self, audio_path, st_time, end_time, bandpass=[1, 10000], classes=['a', 'b', 'c']):
        """
        Initialize the audio bounding box annotator.
        
        Args:
            audio_path (str): Path to the audio file
            st_time (float): Start time in seconds
            end_time (float): End time in seconds
            bandpass (list): Frequency range [min_freq, max_freq] in Hz
            classes (list): List of class names for annotation
        """
        self.audio_path = audio_path
        self.st_time = st_time
        self.end_time = end_time
        self.duration = end_time - st_time
        self.bandpass = bandpass
        self.classes = classes
        
        # Audio and spectrogram data
        self.audio = None
        self.spectrogram = None
        self.spec_data = None
        self.time_axis = None
        self.freq_axis = None
        
        # Image data
        self.image_path = None
        self.image = None
        self.image_width = None
        self.image_height = None
        
        # BBox widget
        self.bbox_widget = None
        
    def load_audio(self):
        """Load audio file with specified parameters."""
        print(f"Loading audio from {self.audio_path}")
        print(f"Time range: {self.st_time}s - {self.end_time}s")
        print(f"Duration: {self.duration}s")
        print(f"Bandpass: {self.bandpass[0]} - {self.bandpass[1]} Hz")
        
        self.audio = Audio.from_file(
            self.audio_path, 
            offset=self.st_time, 
            duration=self.duration
        ).bandpass(self.bandpass[0], self.bandpass[1], order=10)
        
        print(f"Audio loaded successfully. Sample rate: {self.audio.sample_rate} Hz")
        
    def create_spectrogram(self):
        """Create spectrogram and extract axis information."""
        print("Creating spectrogram...")
        
        # Create spectrogram
        self.spectrogram = Spectrogram.from_audio(self.audio)
        
        # Get spectrogram data - use the correct OpenSoundscape API
        self.spec_data = self.spectrogram.spectrogram
        
        # Extract time and frequency axes
        self.time_axis = self.spectrogram.times
        self.freq_axis = self.spectrogram.frequencies
        
        print(f"Spectrogram created with shape: {self.spec_data.shape}")
        print(f"Time axis: {self.time_axis[0]:.2f}s - {self.time_axis[-1]:.2f}s")
        print(f"Frequency axis: {self.freq_axis[0]:.1f} - {self.freq_axis[-1]:.1f} Hz")
        
    def save_spectrogram_image(self, output_path=None, dpi=100, figsize=(12, 8)):
        """Save spectrogram as PNG image with known dimensions."""
        if output_path is None:
            # Create temporary file
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, "temp_spectrogram.png")
        
        print(f"Saving spectrogram to: {output_path}")
        
        # Create figure with specified size
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Plot spectrogram
        im = ax.imshow(
            self.spec_data, 
            aspect='auto', 
            origin='lower',
            extent=[self.time_axis[0], self.time_axis[-1], 
                   self.freq_axis[0], self.freq_axis[-1]]
        )
        
        # Set labels
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(f'Spectrogram: {os.path.basename(self.audio_path)}')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Power (dB)')
        
        # Save with tight layout
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        # Store image information
        self.image_path = output_path
        self.image = Image.open(output_path)
        self.image_width, self.image_height = self.image.size
        
        print(f"Image saved with dimensions: {self.image_width} x {self.image_height} pixels")
        
    def load_image(self):
        """Load the saved spectrogram image."""
        if self.image_path is None:
            raise ValueError("No image path set. Run save_spectrogram_image() first.")
        
        print(f"Loading image from: {self.image_path}")
        self.image = Image.open(self.image_path)
        self.image_width, self.image_height = self.image.size
        print(f"Image loaded with dimensions: {self.image_width} x {self.image_height} pixels")
        
    def create_bbox_widget(self):
        """Create the bounding box widget for annotation."""
        print("Creating bounding box widget...")
        print(f"Available classes: {self.classes}")
        
        # Create BBox widget
        self.bbox_widget = BBoxWidget(
            image=self.image_path,
            classes=self.classes
        )
        
        print("BBox widget created successfully!")
        print("Instructions:")
        print("1. Click and drag to draw bounding boxes")
        print("2. Select a class from the dropdown for each box")
        print("3. Use the widget controls to manage annotations")
        
    def get_annotations_dataframe(self):
        """Convert bounding box annotations to DataFrame with audio coordinates."""
        if self.bbox_widget is None:
            raise ValueError("BBox widget not created. Run create_bbox_widget() first.")
        
        # Get annotations from widget
        annotations = self.bbox_widget.get_annotations()
        
        if not annotations:
            print("No annotations found.")
            return pd.DataFrame(columns=["filepath", "st_time", "end_time", "max_freq", "min_freq", "class"])
        
        # Convert image coordinates to audio coordinates
        rows = []
        for ann in annotations:
            # Extract bounding box coordinates (image space)
            x1, y1, x2, y2 = ann['bbox']
            class_name = ann['class']
            
            # Convert x coordinates (time)
            # x1, x2 are in pixels, need to convert to seconds
            time_start = self.time_axis[0] + (x1 / self.image_width) * (self.time_axis[-1] - self.time_axis[0])
            time_end = self.time_axis[0] + (x2 / self.image_width) * (self.time_axis[-1] - self.time_axis[0])
            
            # Convert y coordinates (frequency)
            # y1, y2 are in pixels, need to convert to Hz
            # Note: y-axis is inverted in image (0 at top, height at bottom)
            freq_max = self.freq_axis[-1] - (y1 / self.image_height) * (self.freq_axis[-1] - self.freq_axis[0])
            freq_min = self.freq_axis[-1] - (y2 / self.image_height) * (self.freq_axis[-1] - self.freq_axis[0])
            
            # Add to absolute time coordinates
            abs_time_start = self.st_time + time_start
            abs_time_end = self.st_time + time_end
            
            rows.append({
                "filepath": self.audio_path,
                "st_time": abs_time_start,
                "end_time": abs_time_end,
                "max_freq": freq_max,
                "min_freq": freq_min,
                "class": class_name
            })
        
        df = pd.DataFrame(rows)
        print(f"Created DataFrame with {len(df)} annotations")
        return df
    
    def run_full_pipeline(self, output_path=None, dpi=100, figsize=(12, 8)):
        """Run the complete pipeline from audio loading to annotation."""
        print("=== Starting Audio BBox Annotation Pipeline ===")
        
        # Step 1: Load audio
        self.load_audio()
        
        # Step 2: Create spectrogram
        self.create_spectrogram()
        
        # Step 3: Save spectrogram image
        self.save_spectrogram_image(output_path, dpi, figsize)
        
        # Step 4: Load image
        self.load_image()
        
        # Step 5: Create BBox widget
        self.create_bbox_widget()
        
        print("=== Pipeline Complete ===")
        print("Use get_annotations_dataframe() to get the final results")
        
        return self.bbox_widget


def create_audio_annotator(audio_path, st_time, end_time, bandpass=[1, 10000], classes=['a', 'b', 'c']):
    """
    Convenience function to create and run the audio annotation pipeline.
    
    Args:
        audio_path (str): Path to the audio file
        st_time (float): Start time in seconds
        end_time (float): End time in seconds
        bandpass (list): Frequency range [min_freq, max_freq] in Hz
        classes (list): List of class names for annotation
    
    Returns:
        AudioBBoxAnnotator: Configured annotator object
    """
    annotator = AudioBBoxAnnotator(audio_path, st_time, end_time, bandpass, classes)
    return annotator 