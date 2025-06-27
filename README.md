# Top Down Listening notebook (Colab)

This contains scripts and a Google Colab notebook for listening to audio clips and labeling them (Top-down listening). This is designed as part of a processing pipeline for data collected using Autonomous recording units (ARUs).

It works by mounting Google Drive into the notebook and loading audio clips. It needs a CSV file containing clip IDs and file paths (in Google Drive) to store input labels. 

For more information, please visit https://www.kitzeslab.org/

## Install as a Python Package

You can now install the annotation tools as a Python package and use them in any project:

```bash
pip install git+https://github.com/LeonardoViotti/tdl-notebook.git
```

Then in your Python code or Jupyter notebook:

```python
from tdl_annotation import annotate, plot_clip

# Use the annotation functions
df = annotate(
    scores_file='your_scores.csv',
    valid_annotations=[0, 1, 2, 3, 4, 'u'],
    skip_cols=['card'],
    n_positives=5,
    index_cols=['file', 'start_time', 'end_time']
)
```

## Dependencies

The package automatically installs these dependencies:
- `opensoundscape>=0.9.1`
- `pandas`
- `numpy`
- `matplotlib`
- `ipython`



## Example Usage

### Basic Usage in a Notebook

```python
# In your annotation_notebook.ipynb
import os
from tdl_annotation import annotate

# Set up your data paths
data_dir = '../data'
scores_file = os.path.join(data_dir, 'scores.csv')

# Start annotation
df = annotate(
    scores_file=scores_file,
    audio_dir=os.path.join(data_dir, 'audio_clips'),
    valid_annotations=[0, 1, 2, 3, 4, 'u'],
    skip_cols=['card'],
    n_positives=5,
    index_cols=['file', 'start_time', 'end_time']
)

print(f"Annotated {len(df)} clips")
```

### Advanced Usage with Custom Parameters

```python
from tdl_annotation import annotate

df = annotate(
    scores_file='clips/_scores-test.csv',
    valid_annotations=[0, 1, 2, 3, 4, 'u'],
    skip_cols=['card'],
    n_positives=5,
    index_cols=['file', 'start_time', 'end_time'],
    dates_filter=['2023-01-01', '2023-01-02'],  # Only annotate specific dates
    card_filter=['card1', 'card2'],              # Only annotate specific cards
    n_sample=100,                                # Sample 100 clips randomly
    dry_run=False,                               # Actually save annotations
    buffer=2.0                                   # Add 2 seconds before/after each clip
)
```

## Notebook set-up

### 1. Google Drive set-up

1. Set up a Google account if you don't already have one.
2. Set up a Google Drive account if you don't already have one
3. Create a folder to be shared with the listener located in the root directory of *My Drive*. It should contain two subdirectories:
```
   └── your-folder-name-here/
     ├── clips/
     └── scripts/
```
4. Upload `annotation.py` scripts to `my-tdl-folder/scripts/`
5. If you already have clips and a scores sheet upload them to `my-tdl-folder/clips/`

IMPORTANT: You can name [*your-folder-name-here*] as you like, as long as that is reflected in the Load data section of the notebook.

### 2. Creating clips and _scores.csv file

The notebook expects an input file named `_scores.csv` **located in the same folder as the clips** containing a row for each clip to be listened to. 

Columns:
 - `relative_path`:  containing paths relative to [*your-folder-name-here*] to audio files.
 - `start_time` (optional): If `_scores.csv` contains paths for longer audio files (not individual clip files), use this to specify the start of each clip.
 - `end_time` (optional): If `_scores.csv` contains paths for longer audio files (not individual clip files), use this to specify the start of each clip.
 - `date` (optional): string dates that can be used to filter rows with `dates_filter` argument (see below.)
 - `card` (optional): recorder SD card id that can be used to filter rows with `card_filter` argument (see below.)
 

At the first run, `tdl_colab.ipynb` will create a copy of `_scores.csv` named `_scores_annotations.csv`. This new file will contain the new columns for annotations and notes and is updated after each clip annotation.

### 3. Set-up notebook

1. Open the [template notebook](https://colab.research.google.com/drive/1Lb288F0hIuYP6L_vUxA2YCaj2OAv8qyS?usp=sharing)
2. File > Save a copy in Google Drive: Select the root directory created.
3. Make any needed changes, e.g. modify annotation options and scores file name.
4. Share the link to the modified notebook with the listener.


Annotation options:

 - `audio_dir (str)`: Directory containing audio clips to be annotated.

 - `valid_annotations`: List of valid options for user. Defaults to ["0", "1", "u"].

 - `scores_filename`: Detection scores CSV filename. This function assumes it is in [audio_dir]. Defaults to "_scores.csv".

 - `annotation_column`: Annotation column name. Defaults to 'annotation'.

 - `dates_filter`: List dates to be annotated (skip others). Defaults to empty list, [].

 - `card_filter`: List cards to be annotated (skip others). Defaults to empty list, [].

 - `skip_cols`: Column names for skipping clips if a positive clip already flagged. 

 - `n_sample`: Sample from valid rows. Defaults to None.

 - `dry_run`:  Not export outputs. Defaults to False.


## Usage instructions

### 1. Google Drive set-up

1. Set up a Google account if you don't already have one.
2. Set up a Google Drive account if you don't already have one. 
3. If you are just listening, share your Google e-mail to receive a shared folder link.
4. Accept the link invitation. The folder named [*your-folder-name-here*] will be located inside the *Shared with me* directory
5. IMPORTANT: follow these steps so the notebook can see the data. It needs the [*your-folder-name-here*] folder to be located in the root directory of *My Drive*.
    - Go to *Shared with me* and click on the *More actions* next to [*your-folder-name-here*].
    - Click *Organize > Add a shortcut*
    - At the top menu select *All locations*
    - Select *My Drive*
    - Click *Add*

IMPORTANT: DO NOT MOVE the shared folder or edit it's contents for the notebook to work.

Notebook usage:
1. At the top menu click *Runtime > Run all*. This will run all the cells in the notebook and install OpenSoundscape (could take a few minutes).
2. After installation, you will be prompted to give the notebook access to your Google Drive files:
    - Connect to Google Drive
    - Choose your account on the pop-up window. If you have more than one Google account, select the one with which the link was shared.
    - Click *Allow*

