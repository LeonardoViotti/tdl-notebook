from opensoundscape.spectrogram import Spectrogram
from opensoundscape.audio import Audio
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
import os
import time

from utils import find_path


def plot_clip(clip_path,  directory = None, bandpass = [1, 10000], mark_at_s = None ):
    """ Load file, display spectograms and play audio

    Args:
        clip_path (str): Audio file path
        directory (str, optional): In case [clip_path] is not a full path, search [directory] for it. Defaults to None.
        bandpass (list, Hz): Length 2 list specifying a frequency range to display. Format is [lower_band, higher_band].
        mark_at_s (list, s): List of seconds to add vertical lines in spectrogram. Typically used to mark start and end of valid clip.
    """
    if directory is not None:
        full_clip_path = find_path(clip_path, directory)
    else:
        full_clip_path = clip_path

    audio = Audio.from_file(full_clip_path).bandpass(bandpass[0],bandpass[1],order=10)
    
    # Add length markings 
    if mark_at_s is not None:
        for s in mark_at_s:
            plt.axvline(x=s, color='b')
    
    ipd.display(Spectrogram.from_audio(audio).bandpass(bandpass[0],bandpass[1]).plot())
    ipd.display(ipd.Audio(audio.samples,rate=audio.sample_rate,autoplay=True))

def user_input(annotations_choices, custom_annotations_dict = None, positive_annotation = '1'):
    """ Request user input given a set of options in [valid_annotations]
    
    Args:
        annotations_choices (list): List of potential annotation choices
        custom_annotations_dict (dictionary, optional): Dictionary containing additional annotation options. Defaults to None.
        positive_annotation (str, optional): Annotation choice that denotes a positive class. Defaults to '1'.

    Returns:
        _type_: _description_
    """
    
    # Make sure all strings are options
    annotations_choices = [str(x).strip().lower() for x in annotations_choices]
    
    # Define paraments to be replaced
    other_annotation = ''
    notes = ''
    
    # Wait for user input within expected parameters
    valid_annotation = False
    while valid_annotation==False:
        annotation = str(input(f"Enter annotation. Valid options are {annotations_choices}.\n").strip()).lower()
        
        if annotation not in annotations_choices:
            print('Not a valid annotation. Please try again.')
            continue
            
        if (annotation==positive_annotation) & (custom_annotations_dict is not None):
            valid_custum_annotation = False
            
            while valid_custum_annotation!=True:
                other_annotation = str(input(f"Add any other annotation? Valid options are {custom_annotations_dict.keys()} or press enter to skip.\n")).lower()
                
                if other_annotation in custom_annotations_dict.keys():
                    other_annotation = custom_annotations_dict[other_annotation]
                    valid_custum_annotation = True

                elif other_annotation=='':
                    custom_annotations_dict = ''
                    valid_custum_annotation = True

                else:
                    print('Not a valid annotation. Please try again.')
                    continue
            
        notes = str(input('Enter any notes you would like to make or press enter to skip.\n'))
        proceed = input(f"Does this look right? Pressing 'r' to try again.\n").lower()
        
        if proceed!='r':
            valid_annotation = True
            
        else:
            continue
    
        return annotation, other_annotation, notes

def save_annotations_file(annotations_df, scores_csv_path):
    """Saves annotations csv at [scores_csv_path] with '_annotations' suffix
    
    Args:
        annotations_df (pd.DataFrame): Clip scores data with annotation columns
        scores_csv_path (str): Path to dave data
    """
    annotations_df.to_csv(f"{scores_csv_path.split('.')[0]}_annotations.csv")

def load_scores_df(scores_csv_path, 
                   annotation_column = 'annotation',
                   index_column = 'relative_path',
                   notes_column = 'notes',
                   custom_annotation_column = 'additional_annotation',
                   sort_by = None, 
                   dry_run = False):
    """Load detection scores CSV data to be annotated. Please refer to README.md for details.
        If it exists, loads partially annotaded data.
    
    Args:
        scores_csv_path (str): Relative or absolute file path to CSV scores data.
        annotation_column (str, optional): Annotation column name. Defaults to 'annotation'.
        index_column (str, optional): Colum to set pd.DataFrame index by. Defaults to 'clip'.
        notes_column (str, optional): Annotation notes column name. Defaults to 'notes'.
        custom_annotation_column (str, optional): If there are custom annotations, column name. Defaults to 'additional_annotation'.
        sort_by (str, optional): Columns to sort scores data by. Defaults to None.
        dry_run (bool, optional): Not export outputs. Defaults to False.
    
    Returns:
        (pd.DataFrame): Data frame containing detection scores.
    """
    # Load or create the annotations csv.
    try:
        scores_df = pd.read_csv(f"{scores_csv_path.split('.')[0]}_annotations.csv")
        scores_df = scores_df.set_index(index_column)
        
        annotation_csv_exists = True
        
    except:
        scores_df = pd.read_csv(scores_csv_path)
        scores_df = scores_df.set_index(index_column)
        
        # Create annotations columns
        scores_df[annotation_column] = np.NaN
        scores_df[notes_column] = np.NaN
        
        if custom_annotation_column is not None:
            scores_df[custom_annotation_column] = np.NaN
        
        if not dry_run: 
            save_annotations_file(scores_df, scores_csv_path)
        
        annotation_csv_exists = False
    
    if sort_by is not None:
        scores_df = scores_df.sort_values('sort_by')
    
    return scores_df, annotation_csv_exists

    
def annotate(audio_dir, 
             valid_annotations = ["0", "1", "u"],
             scores_filename = "_scores.csv", 
             annotation_column = 'annotation',
             path_column = 'relative_path',
             notes_column = 'notes',
             custom_annotation_column = 'additional_annotation',
             skip_if_pos_card_date = False,
             mark_at_s = [0,10],
             sort_by = None, 
             date_filter = [], 
             card_filter = [], 
             custom_annotations_dict = None,
             n_sample = None,
             dry_run = False):
    """Loops through detection scores data that hasn't been annated and aks user to input annotations.

    Args:
        audio_dir (str): Directory containing audio clips to be annotaed.
        valid_annotations (list, optional): List of valid options for user. Defaults to ["0", "1", "u"].
        scores_filename (str, optional): Detection scores CSV filename. This function assumes it is in [audio_dir]. Defaults to "_scores.csv".
        annotation_column (str, optional): Annotation column name. Defaults to 'annotation'.
        dates_filter (list (str), optional): List dates to be annotated (skip others). Defaults to empty list, [].
        card_filter (list (str), optional): List cards to be annotated (skip others). Defaults to empty list, [].
        custom_annotations_dict (dict, optional): _description_. Defaults to None.
        skip_if_pos_card_date (bool, optional): If positive clip already flagged in that date/card skip. 
        n_sample (int, optional): Sample from valid rows. Defaults to None.
        dry_run (bool, optional):  Not export outputs. Defaults to False.
    
    Returns:
        _type_: _description_
    """
    
    scores_csv_path = os.path.join(audio_dir, scores_filename)
    
    scores_df, annotation_csv_exists = load_scores_df(scores_csv_path,
                               annotation_column = annotation_column,
                               index_column = path_column,
                               notes_column = notes_column,
                               custom_annotation_column = custom_annotation_column,
                               sort_by = sort_by, 
                               dry_run = dry_run)
    # scores_df = scores_df.set_index(path_column)
    
    # Skip of data ot card filter provided
    if date_filter or card_filter:
        scores_df['skip'] = (scores_df['date'].isin(date_filter)) | (scores_df['card'].isin(card_filter))
    else:
        scores_df['skip'] = False
    
    # Skip if skip_if_present columns provided.
    
    # Create absolute path index
    # scores_df['absolute_path'] = audio_dir + '/' + scores_df['relative_path']
    scores_df['absolute_path'] = audio_dir + '/' + scores_df.index
    
    # scores_df = scores_df.set_index('absolute_path')
    
    valid_rows = scores_df[~scores_df[annotation_column].notnull()]
    if n_sample is not None:
        valid_rows = valid_rows.sample(n_sample)
    
    # Print total variables
    n_clips = len(scores_df)
    n_clips_remaining = len(valid_rows)
    n_skiped_clips = sum(scores_df['skip'])
    n_clips_filtered = n_clips - n_skiped_clips
    
    # for idx,row in valid_rows.iterrows():
    while len(valid_rows) > 0:
        row = valid_rows.iloc[0]
        idx = valid_rows.index[0]
    
        # Clear previous plot if any
        ipd.clear_output(wait = True)
        
        # Print progress
        annotated_total = n_clips - n_clips_remaining
        annotated_not_skiped = sum(scores_df[annotation_column].notnull() & scores_df[annotation_column].isin(valid_annotations))
        if (not date_filter) & (not card_filter):
            print(f'{annotated_total} of {n_clips}')
        else:
            print(f'{annotated_not_skiped} of {n_clips_filtered}')
        
        # Annotate
        if row['skip']:
            scores_df.at[idx, annotation_column] = "not reviewed"
        else:
            print(f"Clip: {idx}")
        
            # plot_clip(idx, mark_at_s = [3, 7])
            plot_clip(row['absolute_path'], mark_at_s = mark_at_s)
            time.sleep(.1) # Added delay for stability (hopefully)
            annotations = user_input(valid_annotations, custom_annotations_dict = custom_annotations_dict, positive_annotation = '1')

            scores_df.at[idx, annotation_column] = annotations[0]
            scores_df.at[idx, custom_annotation_column] = annotations[1]
            scores_df.at[idx, notes_column]= annotations[2]
            
            if skip_if_pos_card_date:
                print('Skipping date and card')
                assert set(['card','date']).issubset(scores_df.columns), "'card' and/or 'date' colums not present."
                if scores_df.at[idx, annotation_column] == '1':
                    card_i = row['card']
                    date_i = row['date']
                    skip_condition = (scores_df['card'] == card_i) & (scores_df['date'] == date_i) & (scores_df[annotation_column].isna())
                    scores_df.loc[skip_condition,annotation_column] = 'skipped'
        
        # print(scores_df)
        # time.sleep(10) # Added delay for stability (hopefully)
        
        if not dry_run: 
            # save_annotations_file(scores_df.drop(['skip', 'absolute_path'], axis = 1), scores_csv_path)
            # scores_df = scores_df.reset_index()
            save_annotations_file(scores_df.drop(['absolute_path'], axis = 1), scores_csv_path)
        
        # Update params
        n_clips_remaining = len(scores_df[~scores_df[annotation_column].notnull()])
        valid_rows = scores_df[~scores_df[annotation_column].notnull()]
    
    return scores_df
