from opensoundscape.spectrogram import Spectrogram
from opensoundscape.audio import Audio
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
import os
import time
import tempfile
from PIL import Image


#-----------------------------------------------------------------------------------------------------
# Functions


def plot_clip(audio_path,
              st = None,
              end = None,
              bandpass = [1, 10000], 
              mark_at_s = None,
              buffer = None):
    """ Load file, display spectograms and play audio
    
    Args:
        audio_path (str): Audio file path
        directory (str, optional): In case [audio_path] is not a full path, search [directory] for it. Defaults to None.
        bandpass (list, Hz): Length 2 list specifying a frequency range to display. Format is [lower_band, higher_band].
        mark_at_s (list, s): List of seconds to add vertical lines in spectrogram. Typically used to mark start and end of valid clip.
        buffer (float, optional): Add buffer to beginning and end of clip (seconds). Defaults to None.
    """
    
    # Apply buffer if provided
    if buffer and st is not None and end is not None:
        st_buffered = max(0, st - buffer)
        end_buffered = end + buffer
        dur = end_buffered - st_buffered
        audio = Audio.from_file(audio_path, offset=st_buffered, duration=dur).bandpass(bandpass[0], bandpass[1], order=10)
        
        # Automatically mark original clip boundaries when buffer is used
        if mark_at_s is None:
            # Calculate relative positions of original boundaries within buffered audio
            original_st_relative = st - st_buffered
            original_end_relative = end - st_buffered
            mark_at_s = [original_st_relative, original_end_relative]
        else:
            # If mark_at_s is provided, adjust it relative to the buffered start
            mark_at_s = [m - st_buffered for m in mark_at_s]
    else:
        # Original behavior
        dur = end - st
        audio = Audio.from_file(audio_path, offset=st, duration=dur).bandpass(bandpass[0], bandpass[1], order=10)
    
    # Add length markings 
    if mark_at_s is not None:
        for s in mark_at_s:
            plt.axvline(x=s, color='b')
    
    ipd.display(Spectrogram.from_audio(audio).bandpass(bandpass[0], bandpass[1]).plot())
    ipd.display(ipd.Audio(audio.samples, rate=audio.sample_rate, autoplay=True))

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
                   index_cols = 'relative_path',
                   notes_column = 'notes',
                   custom_annotation_column = None,
                   sort_by = None, 
                   dry_run = False):
    """Load detection scores CSV data to be annotated. Please refer to README.md for details.
        If it exists, loads partially annotaded data.
    
    Args:
        scores_csv_path (str): Relative or absolute file path to CSV scores data.
        annotation_column (str, optional): Annotation column name. Defaults to 'annotation'.
        index_cols (str, optional): Colum to set pd.DataFrame index by. Defaults to 'clip'.
        notes_column (str, optional): Annotation notes column name. Defaults to 'notes'.
        custom_annotation_column (str, optional): If there are custom annotations, column name. Defaults to 'additional_annotation'.
        sort_by (list, optional): Columns to sort scores data by. Defaults to None.
        dry_run (bool, optional): Not export outputs. Defaults to False.
    
    Returns:
        (pd.DataFrame): Data frame containing detection scores.
    """
    # Load or create the annotations csv.
    try:
        scores_df = pd.read_csv(f"{scores_csv_path.split('.')[0]}_annotations.csv")
        scores_df = scores_df.set_index(index_cols)
        
        annotation_csv_exists = True
        
    except:
        scores_df = pd.read_csv(scores_csv_path)
        scores_df = scores_df.set_index(index_cols)
        
        # Create annotations columns
        scores_df[annotation_column] = np.NaN
        scores_df[notes_column] = np.NaN
        
        # # Testar essa porra
        # scores_df['num_annotation'] = 0
        # scores_df['cum_sum'] = 0
        
        if custom_annotation_column:
            scores_df[custom_annotation_column] = np.NaN
            # for col in custom_annotation_columns:
            #     scores_df[col] = np.NaN
        
        if not dry_run: 
            save_annotations_file(scores_df, scores_csv_path)
        
        annotation_csv_exists = False
    
    if sort_by is not None:
        scores_df = scores_df.sort_values('sort_by')
    
    return scores_df, annotation_csv_exists


def annotate(scores_file = "_scores.csv",
             audio_dir = None, 
             valid_annotations = ["0", "1", "u"],
             annotation_column = 'annotation',
            #  path_column = 'relative_path',
             index_cols = ['relative_path'],
             notes_column = 'notes',
             custom_annotation_column = 'additional_annotation',
             skip_cols = None,
             n_positives = 1,
             n_negatives = 100,
             mark_at_s = None,
             sort_by = None, 
             date_filter = [], 
             card_filter = [], 
             custom_annotations_dict = None,
             n_sample = None,
             dry_run = False,
             buffer = None):
    """Loops through detection scores data that hasn't been annated and aks user to input annotations.
    
    Args:
        scores_file (str, optional): Detection scores CSV file path or filename if it is in [audio_dir]. Defaults to "_scores.csv".
        audio_dir (str): If using relative file paths and [scores_file] in [audio_dir]. It should be a directory containing audio clips to be annotated.  Defaults to None.
        valid_annotations (list, optional): List of valid options for user. Defaults to ["0", "1", "u"].
        annotation_column (str, optional): Annotation column name. Defaults to 'annotation'.
        index_cols (str, optional): Either ["path_to_clip"] or ["path_to_audio", "clip_start_time", "clip_end_time"]. Defaults to 'relative_path'.
        notes_column (str, optional): Column name for notes. Defaults to 'notes'.
        custom_annotation_column (str, optional): Column name for additional annotation. Defaults to 'additional_annotation'.
        skip_cols (str, optional): Column names for skipping clips if a positive clip already flagged.
        n_positives (int, optional): Number of positives needed before skipping if skip_cols is provided. Defaults to None.
        n_negatives (int, optional): Number of negatives needed before skipping if skip_cols is provided. Defaults to 100.
        mark_at_s (list, optional): Seconds to mark clip with vertical lines. Usually to define start and end of clip if there is padding. Defaults to None.
        sort_by (list, optional): Columns to sort scores data by. Defaults to None.
        date_filter (list (str), optional): List dates to be annotated (skip others). Defaults to empty list, [].
        card_filter (list (str), optional): List cards to be annotated (skip others). Defaults to empty list, [].
        custom_annotations_dict (dict, optional): _description_. Defaults to None.
        n_sample (int, optional): Sample from valid rows. Defaults to None.
        dry_run (bool, optional):  Not export outputs. Defaults to False.
        buffer (float, optional): Add buffer to beginning and end of clip (seconds). Defaults to None.
    
    Exports:
        Every iteration exports a file named [scores_file]_annotations.csv to [audio_dir] 
        
    Returns:
        pd.DataFrame with annotations
    """
    
    # If using relative file paths
    if audio_dir:
        scores_csv_path = os.path.join(audio_dir, scores_file)
    else:
        scores_csv_path = scores_file
    
    
    scores_df, annotation_csv_exists = load_scores_df(scores_csv_path,
                               annotation_column = annotation_column,
                               index_cols = index_cols,
                               notes_column = notes_column,
                               custom_annotation_column = custom_annotation_column,
                               sort_by = sort_by, 
                               dry_run = dry_run)
    
    # Add placeholder for skip intermediate columns (won't be exported)
    if skip_cols:
        scores_df['num_annotation'] = np.NaN
        scores_df['cum_sum'] = np.NaN
        scores_df['num_negatives'] = np.NaN
        scores_df['cum_sum_negatives'] = np.NaN
    
    # Skip of data ot card filter provided
    if date_filter or card_filter:
        scores_df['skip'] = (scores_df['date'].isin(date_filter)) | (scores_df['card'].isin(card_filter))
    else:
        scores_df['skip'] = False
    
    # Skip if skip_if_present columns provided.
    valid_rows = scores_df[~scores_df[annotation_column].notnull()]
    if n_sample is not None:
        valid_rows = valid_rows.sample(n_sample)
    
    # Print total variables
    n_clips = len(scores_df)
    n_clips_remaining = len(valid_rows)
    n_skiped_clips = sum(scores_df['skip'])
    n_clips_filtered = n_clips - n_skiped_clips
    
    # Placeholder for cumulative sum of positives
    current_cum_sum = None
    
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
            
            if current_cum_sum is not None:
                print(f'{current_cum_sum.item()} positives out of {n_positives} for this ' + f'{" and ".join(str(col) for col in skip_cols)}')
            
            if len(index_cols) == 1: # Assume it is a path for an already trimed clip
                plot_clip(idx, mark_at_s = mark_at_s, buffer = buffer)
            elif len(index_cols) == 3:
                plot_clip(idx[0], idx[1], idx[2], mark_at_s = mark_at_s, buffer = buffer)
            else:
                raise Exception('index_cols must be either ["path_to_clip"] or ["path_to_audio", "clip_start_time", "clip_end_time"]')
            
            time.sleep(.1) # Added delay for stability (hopefully)
            annotations = user_input(valid_annotations, 
                                     custom_annotations_dict = custom_annotations_dict, positive_annotation = '1')

            scores_df.loc[idx, annotation_column] = annotations[0]
            scores_df.loc[idx, custom_annotation_column] = annotations[1]
            scores_df.loc[idx, notes_column]= annotations[2]
            
            if skip_cols:
                assert set(skip_cols).issubset(scores_df.columns), "skip_cols not present!"
                assert isinstance(skip_cols, list), f'skip_cols argument must be a list!'
                
                # Update the cumulative sum every iteration
                
                # Column that counts all annotations greater than 0
                scores_df['num_annotation'] =  (pd.to_numeric(scores_df[annotation_column], errors='coerce').fillna(0) > 0).astype(int)
                # scores_df['num_annotation'] =  (~scores_df[annotation_column].isna()).astype(int)
                
                # Column that counts all negative annoations
                scores_df['num_negatives'] =  (pd.to_numeric(scores_df[annotation_column], errors='coerce').fillna(0) == 0).astype(int)

                # Cumulative sums coluymns
                scores_df['cum_sum'] = scores_df.groupby(skip_cols)['num_annotation'].cumsum()
                scores_df['cum_sum_negatives'] = scores_df.groupby(skip_cols)['num_negatives'].cumsum()
                
                current_cum_sum = scores_df.loc[idx, 'cum_sum']
                current_cum_negatives = scores_df.loc[idx, 'cum_sum_negatives']
                if ((current_cum_sum >= n_positives) | (current_cum_negatives >= n_negatives)).item():
                    
                    # Create bolean series if row equal to current value of skip col
                    skip_bool_list = []
                    for skip_col in skip_cols:
                        skip_value = row[skip_col]
                        skip_bool = scores_df[skip_col] == skip_value
                        skip_bool_list.append(skip_bool)
                    
                    # Add condition to skip that it cannot already be annotated
                    skip_bool_list.append(scores_df[annotation_column].isna())
                    
                    # Collapse bool series if value is the same as all skip coluns
                    skip_bool_series = pd.concat(skip_bool_list, axis=1).all(axis=1)
                    
                    scores_df.loc[skip_bool_series,annotation_column] = 'skipped'
                
        if not dry_run: 
            if skip_cols:
                save_annotations_file(scores_df.drop(['skip', 'num_annotation', 'cum_sum', 'num_negatives', 'cum_sum_negatives'], axis = 1), scores_csv_path)
            else:
                save_annotations_file(scores_df.drop(['skip'], axis = 1), scores_csv_path)
            # save_annotations_file(scores_df, scores_csv_path)
        # Update params
        n_clips_remaining = len(scores_df[~scores_df[annotation_column].notnull()])
        valid_rows = scores_df[~scores_df[annotation_column].notnull()]
    
    return scores_df

def annotate_bbox(scores_file = "_scores.csv",
                  audio_dir = None, 
                  valid_annotations = ["0", "1", "u"],
                  annotation_column = 'annotation',
                  index_cols = ['relative_path'],
                  notes_column = 'notes',
                  custom_annotation_column = 'additional_annotation',
                  bbox_columns = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'],
                  mark_at_s = None,
                  sort_by = None, 
                  date_filter = [], 
                  card_filter = [], 
                  custom_annotations_dict = None,
                  n_sample = None,
                  dry_run = False,
                  buffer = None):
    """Loops through detection scores data and allows interactive drawing of bounding boxes on spectrograms.
    
    Args:
        scores_file (str, optional): Detection scores CSV file path or filename if it is in [audio_dir]. Defaults to "_scores.csv".
        audio_dir (str): If using relative file paths and [scores_file] in [audio_dir]. It should be a directory containing audio clips to be annotated.  Defaults to None.
        valid_annotations (list, optional): List of valid options for user. Defaults to ["0", "1", "u"].
        annotation_column (str, optional): Annotation column name. Defaults to 'annotation'.
        index_cols (str, optional): Either ["path_to_clip"] or ["path_to_audio", "clip_start_time", "clip_end_time"]. Defaults to 'relative_path'.
        notes_column (str, optional): Column name for notes. Defaults to 'notes'.
        custom_annotation_column (str, optional): Column name for additional annotation. Defaults to 'additional_annotation'.
        bbox_columns (list, optional): Column names for bounding box coordinates [x1, y1, x2, y2]. Defaults to ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'].
        mark_at_s (list, optional): Seconds to mark clip with vertical lines. Usually to define start and end of clip if there is padding. Defaults to None.
        sort_by (list, optional): Columns to sort scores data by. Defaults to None.
        date_filter (list (str), optional): List dates to be annotated (skip others). Defaults to empty list, [].
        card_filter (list (str), optional): List cards to be annotated (skip others). Defaults to empty list, [].
        custom_annotations_dict (dict, optional): Dictionary containing additional annotation options. Defaults to None.
        n_sample (int, optional): Sample from valid rows. Defaults to None.
        dry_run (bool, optional): Not export outputs. Defaults to False.
        buffer (float, optional): Add buffer to beginning and end of clip (seconds). Defaults to None.
    
    Exports:
        Every iteration exports a file named [scores_file]_bbox_annotations.csv to [audio_dir] 
        
    Returns:
        pd.DataFrame with bounding box annotations
    """
    
    try:
        from jupyter_bbox_widget import BBoxWidget
    except ImportError:
        print("jupyter_bbox_widget not found. Please install it with: pip install jupyter-bbox-widget")
        return None
    
    # If using relative file paths
    if audio_dir:
        scores_csv_path = os.path.join(audio_dir, scores_file)
    else:
        scores_csv_path = scores_file
    
    # Load the scores dataframe
    scores_df = pd.read_csv(scores_csv_path)
    scores_df = scores_df.set_index(index_cols)
    
    # Create bbox annotations dataframe
    bbox_annotations_path = f"{scores_csv_path.split('.')[0]}_bbox_annotations.csv"
    
    try:
        bbox_df = pd.read_csv(bbox_annotations_path)
        bbox_df = bbox_df.set_index(index_cols)
        bbox_annotations_exist = True
    except:
        # Create empty dataframe with proper columns
        bbox_df = pd.DataFrame(columns=bbox_columns + [annotation_column, custom_annotation_column, notes_column])
        bbox_annotations_exist = False
    
    # Skip of data or card filter provided
    if date_filter or card_filter:
        scores_df['skip'] = (scores_df['date'].isin(date_filter)) | (scores_df['card'].isin(card_filter))
    else:
        scores_df['skip'] = False
    
    # Get valid rows (not skipped)
    valid_rows = scores_df[~scores_df['skip']]
    if n_sample is not None:
        valid_rows = valid_rows.sample(n_sample)
    
    # Print total variables
    n_clips = len(scores_df)
    n_clips_filtered = len(valid_rows)
    
    print(f"Starting interactive bbox annotation for {n_clips_filtered} clips")
    
    # Loop through clips
    for i, (idx, row) in enumerate(valid_rows.iterrows()):
        # Clear previous plot if any
        ipd.clear_output(wait=True)
        
        # Print progress
        print(f'Clip {i+1} of {n_clips_filtered}: {idx}')
        
        # Load audio and create spectrogram
        if len(index_cols) == 1:  # Assume it is a path for an already trimmed clip
            audio_path = idx
            st, end = None, None
        elif len(index_cols) == 3:
            audio_path = idx[0]
            st, end = idx[1], idx[2]
        else:
            raise Exception('index_cols must be either ["path_to_clip"] or ["path_to_audio", "clip_start_time", "clip_end_time"]')
        
        # Apply buffer if provided
        if buffer and st is not None and end is not None:
            st_buffered = max(0, st - buffer)
            end_buffered = end + buffer
            dur = end_buffered - st_buffered
            audio = Audio.from_file(audio_path, offset=st_buffered, duration=dur)
        else:
            if st is not None and end is not None:
                dur = end - st
                audio = Audio.from_file(audio_path, offset=st, duration=dur)
            else:
                audio = Audio.from_file(audio_path)
        
        # Create spectrogram
        spec = Spectrogram.from_audio(audio)
        
        # Convert spectrogram to image for the widget
        # Create a temporary image file from the spectrogram
        import tempfile
        import matplotlib.pyplot as plt
        
        # Create spectrogram plot with proper settings
        plt.figure(figsize=(12, 8))
        spec.plot()
        plt.title(f"Clip: {idx}")
        
        # Add length markings if provided
        if mark_at_s is not None:
            for s in mark_at_s:
                plt.axvline(x=s, color='b', linestyle='--')
        
        # Get the current axes to understand the coordinate system
        ax = plt.gca()
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        
        # Debug: print spectrogram info
        try:
            spec_data = spec.values
            print(f"Spectrogram shape: {spec_data.shape}")
        except AttributeError:
            try:
                spec_data = spec.data
                print(f"Spectrogram shape: {spec_data.shape}")
            except AttributeError:
                spec_data = np.array(spec)
                print(f"Spectrogram shape: {spec_data.shape}")
        
        print(f"Spectrogram time range: {spec.times[0]:.2f} to {spec.times[-1]:.2f} seconds")
        print(f"Spectrogram frequency range: {spec.frequencies[0]:.2f} to {spec.frequencies[-1]:.2f} Hz")
        print(f"Plot axes limits - X: {x_lim}, Y: {y_lim}")
        
        # Save to temporary file with high quality
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            plt.savefig(tmp_file.name, bbox_inches='tight', pad_inches=0.1, dpi=150, facecolor='white')
            image_path = tmp_file.name
        
        plt.close()
        
        # Verify the image was created and get its dimensions
        try:
            img = Image.open(image_path)
            print(f"Saved image dimensions: {img.size[0]} x {img.size[1]} pixels")
            print(f"Image file path: {image_path}")
        except Exception as e:
            print(f"Error reading saved image: {e}")
        
        # Display spectrogram with bbox widget
        print("Draw bounding boxes on the spectrogram below:")
        print("Instructions:")
        print("- Click and drag to draw rectangles")
        print("- Double-click to delete a box")
        print("- Press 'Done' when finished drawing")
        print(f"Coordinate system: X (time): {x_lim[0]:.2f} to {x_lim[1]:.2f} seconds")
        print(f"Coordinate system: Y (frequency): {y_lim[0]:.2f} to {y_lim[1]:.2f} Hz")
        
        # Create bbox widget
        bbox_widget = BBoxWidget(
            image=image_path,  # The spectrogram image file path
            classes=[str(x) for x in valid_annotations]  # Convert all to strings
        )
        
        # Display the widget
        display(bbox_widget)
        
        # Wait for user to finish drawing
        print("\nDraw your bounding boxes above, then press Enter to continue...")
        input()
        
        # Get the drawn boxes
        bboxes = bbox_widget.bboxes
        
        if not bboxes:
            print("No bounding boxes drawn. Skipping this clip.")
            continue
        
        print(f"Found {len(bboxes)} bounding boxes. Annotating each one...")
        
        # For each bounding box, get annotation
        for bbox_idx, bbox in enumerate(bboxes):
            print(f"\nAnnotating bounding box {bbox_idx + 1} of {len(bboxes)}")
            
            # Convert widget coordinates to spectrogram coordinates
            # Widget coordinates are in pixels, need to convert to spectrogram coordinates
            img_width = bbox_widget.image_width
            img_height = bbox_widget.image_height
            
            print(f"Widget image dimensions: {img_width} x {img_height} pixels")
            print(f"Widget box coordinates: x={bbox['x']:.1f}, y={bbox['y']:.1f}, w={bbox['width']:.1f}, h={bbox['height']:.1f}")
            
            # Convert from widget pixel coordinates to spectrogram coordinates
            x1 = x_lim[0] + (bbox['x'] / img_width) * (x_lim[1] - x_lim[0])
            y1 = y_lim[0] + (bbox['y'] / img_height) * (y_lim[1] - y_lim[0])
            x2 = x_lim[0] + ((bbox['x'] + bbox['width']) / img_width) * (x_lim[1] - x_lim[0])
            y2 = y_lim[0] + ((bbox['y'] + bbox['height']) / img_height) * (y_lim[1] - y_lim[0])
            
            print(f"Converted coordinates: x1={x1:.2f}s, y1={y1:.2f}Hz, x2={x2:.2f}s, y2={y2:.2f}Hz")
            
            # Get annotation for this bounding box
            annotation, other_annotation, notes = user_input(
                valid_annotations, 
                custom_annotations_dict=custom_annotations_dict, 
                positive_annotation='1'
            )
            
            # Add to dataframe
            bbox_row = pd.DataFrame({
                bbox_columns[0]: [x1],
                bbox_columns[1]: [y1], 
                bbox_columns[2]: [x2],
                bbox_columns[3]: [y2],
                annotation_column: [annotation],
                custom_annotation_column: [other_annotation],
                notes_column: [notes]
            }, index=[idx])
            
            bbox_df = pd.concat([bbox_df, bbox_row])
        
        # Save after each clip
        if not dry_run:
            bbox_df.to_csv(bbox_annotations_path)
            print(f"Saved {len(bbox_df)} bounding box annotations to {bbox_annotations_path}")
        
        # Ask if user wants to continue
        continue_annotation = input("\nPress Enter to continue to next clip, or 'q' to quit: ")
        if continue_annotation.lower() == 'q':
            break
    
    print(f"\nAnnotation complete! Total bounding boxes: {len(bbox_df)}")
    return bbox_df



