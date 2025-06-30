"""
Interactive Bounding Box Annotator using jupyter-bbox-widget

This script uses the jupyter-bbox-widget library to create interactive
bounding box annotation on spectrograms.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Audio processing imports
from opensoundscape.spectrogram import Spectrogram
from opensoundscape.audio import Audio
import IPython.display as ipd

# BBox widget import
from jupyter_bbox_widget import BBoxWidget


class SpectrogramBBoxAnnotator:
    """
    Interactive bounding box annotator for spectrograms using jupyter-bbox-widget.
    """
    
    def __init__(self, audio_path, st=None, end=None, bandpass=[1, 10000], 
                 mark_at_s=None, buffer=None, classes=['call', 'noise', 'background', 'other']):
        """
        Initialize the annotator.
        
        Args:
            audio_path (str): Path to audio file
            st (float): Start time in seconds
            end (float): End time in seconds
            bandpass (list): Frequency range [low, high] in Hz
            mark_at_s (list): Times to mark with vertical lines
            buffer (float): Buffer time in seconds
            classes (list): List of annotation class names
        """
        self.audio_path = audio_path
        self.st = st
        self.end = end
        self.bandpass = bandpass
        self.mark_at_s = mark_at_s
        self.buffer = buffer
        self.classes = classes
        
        # Load audio and create spectrogram
        self._load_audio()
        self._create_spectrogram()
        
    def _load_audio(self):
        """Load and process audio file."""
        if self.buffer and self.st is not None and self.end is not None:
            st_buffered = max(0, self.st - self.buffer)
            end_buffered = self.end + self.buffer
            dur = end_buffered - st_buffered
            self.audio = Audio.from_file(self.audio_path, offset=st_buffered, duration=dur).bandpass(
                self.bandpass[0], self.bandpass[1], order=10)
            
            if self.mark_at_s is None:
                original_st_relative = self.st - st_buffered
                original_end_relative = self.end - st_buffered
                self.mark_at_s = [original_st_relative, original_end_relative]
            else:
                self.mark_at_s = [m - st_buffered for m in self.mark_at_s]
        else:
            dur = self.end - self.st
            self.audio = Audio.from_file(self.audio_path, offset=self.st, duration=dur).bandpass(
                self.bandpass[0], self.bandpass[1], order=10)
    
    def _create_spectrogram(self):
        """Create spectrogram from audio."""
        self.spec = Spectrogram.from_audio(self.audio).bandpass(self.bandpass[0], self.bandpass[1])
        self.spec_array = self.spec.spectrogram
        self.times = self.spec.times
        self.freqs = self.spec.frequencies
    
    def create_spectrogram_image(self, save_path='temp_spectrogram.png'):
        """Create and save spectrogram as image for the bbox widget."""
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot spectrogram
        im = ax.imshow(self.spec_array, aspect='auto', origin='lower', 
                      extent=[self.times[0], self.times[-1], self.freqs[0], self.freqs[-1]])
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Power')
        
        # Add vertical lines for marks
        if self.mark_at_s is not None:
            for s in self.mark_at_s:
                ax.axvline(x=s, color='blue', linestyle='--', alpha=0.7)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(f'Spectrogram Annotation - {self.audio_path.split("/")[-1]}')
        
        plt.tight_layout()
        
        # Save the image
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_bbox_widget(self, image_path='temp_spectrogram.png'):
        """Create the bbox widget with the spectrogram image."""
        # Create the bbox widget
        bbox_widget = BBoxWidget(
            image=image_path,
            classes=self.classes
        )
        
        return bbox_widget
    
    def get_annotations_df(self, bbox_widget):
        """Convert bbox widget annotations to DataFrame."""
        annotations = []
        
        for i, bbox in enumerate(bbox_widget.bboxes):
            # Convert pixel coordinates to time/frequency
            x1, y1, x2, y2 = bbox['x'], bbox['y'], bbox['x'] + bbox['width'], bbox['y'] + bbox['height']
            
            # Convert to time and frequency (assuming linear mapping)
            time_start = self.times[0] + (x1 / bbox_widget.image_width) * (self.times[-1] - self.times[0])
            time_end = self.times[0] + (x2 / bbox_widget.image_width) * (self.times[-1] - self.times[0])
            
            # Note: y-coordinates are inverted in image space
            freq_start = self.freqs[0] + ((bbox_widget.image_height - y2) / bbox_widget.image_height) * (self.freqs[-1] - self.freqs[0])
            freq_end = self.freqs[0] + ((bbox_widget.image_height - y1) / bbox_widget.image_height) * (self.freqs[-1] - self.freqs[0])
            
            annotation = {
                'start_time': time_start,
                'end_time': time_end,
                'start_freq': freq_start,
                'end_freq': freq_end,
                'class': bbox['label'],
                'box_id': i
            }
            annotations.append(annotation)
        
        return pd.DataFrame(annotations)
    
    def display_interface(self):
        """Display the complete annotation interface."""
        # Create spectrogram image
        image_path = self.create_spectrogram_image()
        
        # Create bbox widget
        bbox_widget = self.create_bbox_widget(image_path)
        
        # Display the widget
        display(bbox_widget)
        
        print("\nInstructions:")
        print("1. Click and drag to draw bounding boxes")
        print("2. Select a class from the dropdown for each box")
        print("3. Use the widget controls to manage annotations")
        print("4. Use get_annotations_df(bbox_widget) to get annotations as DataFrame")
        
        return bbox_widget


def create_bbox_widget_annotator(audio_path, st=None, end=None, bandpass=[1, 10000], 
                                mark_at_s=None, buffer=None, classes=['call', 'noise', 'background', 'other']):
    """
    Convenience function to create and display a bbox widget annotator.
    
    Args:
        audio_path (str): Path to audio file
        st (float): Start time in seconds
        end (float): End time in seconds
        bandpass (list): Frequency range [low, high] in Hz
        mark_at_s (list): Times to mark with vertical lines
        buffer (float): Buffer time in seconds
        classes (list): List of annotation class names
    
    Returns:
        tuple: (SpectrogramBBoxAnnotator, BBoxWidget)
    """
    annotator = SpectrogramBBoxAnnotator(
        audio_path=audio_path,
        st=st,
        end=end,
        bandpass=bandpass,
        mark_at_s=mark_at_s,
        buffer=buffer,
        classes=classes
    )
    
    bbox_widget = annotator.display_interface()
    return annotator, bbox_widget 