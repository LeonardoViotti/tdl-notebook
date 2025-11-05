from opensoundscape.spectrogram import Spectrogram
from opensoundscape.audio import Audio
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
import os
import time


# ----------------------------------------------------------------------------------
# plot_clip_mixit
# ----------------------------------------------------------------------------------
def plot_clip_mixit(audio_path,
                    st=None,
                    end=None,
                    bandpass=[1, 10000],
                    mark_at_s=None,
                    buffer=None,
                    model=None,
                    max_sources=8,
                    play_audio=True,
                    window_samples=200,
                    cmap="Greys",
                    vmin=-100,
                    vmax=-20):
    """
    Like plot_clip(), but when a separation `model` is provided it also plots separated sources'
    spectrograms (adaptive grid, commonly 4 or 8) and plays the buffered/original audio.

    Args:
        audio_path (str): Audio file path (or whatever `plot_clip` expects as first arg).
        st (float): clip start (s).
        end (float): clip end (s).
        bandpass (list): [low_hz, high_hz] applied to audio/spectrogram.
        mark_at_s (list): seconds at which to draw vertical markers (relative to plotted audio).
        buffer (float): seconds to expand clip on each side (applied before separation/plotting).
        model (optional): object with `separate_audio(audio)` or `separate(audio)` returning an iterable
                          of sources (preferably opensoundscape.Audio objects).
        max_sources (int): maximum number of separated sources to plot (truncate if model returns more).
        play_audio (bool): whether to play/display the (buffered) combined audio at the end.
        window_samples (int): passed to Spectrogram.from_audio when computing spectrograms.
        cmap, vmin, vmax: visualization parameters for pcolormesh.
    """
    # --- load / apply buffer (same logic as original plot_clip) ---
    if buffer and st is not None and end is not None:
        st_buffered = max(0, st - buffer)
        end_buffered = end + buffer
        dur = end_buffered - st_buffered
        audio = Audio.from_file(audio_path, offset=st_buffered, duration=dur).bandpass(bandpass[0], bandpass[1], order=10)
        # adjust mark positions to buffered-relative coordinates
        if mark_at_s is None:
            original_st_relative = st - st_buffered
            original_end_relative = end - st_buffered
            mark_at_s = [original_st_relative, original_end_relative]
        else:
            mark_at_s = [m - st_buffered for m in mark_at_s]
    else:
        # original behavior
        dur = end - st
        audio = Audio.from_file(audio_path, offset=st, duration=dur).bandpass(bandpass[0], bandpass[1], order=10)

    # If no model provided, behave like original plot_clip
    if model is None:
        ipd.display(Spectrogram.from_audio(audio).bandpass(bandpass[0], bandpass[1]).plot())
        if play_audio:
            ipd.display(ipd.Audio(audio.samples, rate=audio.sample_rate, autoplay=True))
        return

    # --- attempt to call model separation API (be defensive) ---
    try:
        separated = model.separate_audio(audio)
    except Exception:
        try:
            separated = model.separate(audio)
        except Exception as e:
            raise RuntimeError("Model separation failed: ensure model has `separate_audio` or `separate` method.") from e

    # --- normalize separated outputs into a list of opensoundscape.Audio objects ---
    clip_separated = []
    for src in separated:
        if isinstance(src, Audio):
            clip_separated.append(src)
        elif isinstance(src, (tuple, list)) and len(src) >= 2 and isinstance(src[0], (np.ndarray, list)):
            # treat as (samples, sample_rate) or similar
            samples = np.asarray(src[0])
            sr = int(src[1])
            clip_separated.append(Audio(samples=samples, sample_rate=sr))
        elif isinstance(src, np.ndarray):
            # samples only
            clip_separated.append(Audio(samples=src, sample_rate=audio.sample_rate))
        else:
            # try duck-typing: objects with .samples and .sample_rate
            samples = getattr(src, "samples", None)
            sr = getattr(src, "sample_rate", None)
            if samples is not None and sr is not None:
                clip_separated.append(Audio(samples=np.asarray(samples), sample_rate=int(sr)))
            else:
                raise TypeError(f"Unknown separated source type: {type(src)}. Expected Audio, (samples,sr), or ndarray.")

    # Truncate if model returned more than allowed
    total_sources = len(clip_separated)
    n_sources = min(total_sources, max_sources)
    if total_sources > n_sources:
        print(f"Model returned {total_sources} sources; plotting first {n_sources}. Increase max_sources to show more.")

    if n_sources == 0:
        print("Model returned no separated sources to plot.")
        return

    # --- compute grid layout ---
    n_cols = int(np.ceil(np.sqrt(n_sources)))
    n_rows = int(np.ceil(n_sources / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    # normalize axes to flat list for consistent indexing
    if isinstance(axes, np.ndarray):
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    # --- plot each separated source spectrogram ---
    for i in range(n_sources):
        src_audio = clip_separated[i]
        spec = Spectrogram.from_audio(src_audio, window_samples=window_samples).bandpass(bandpass[0], bandpass[1])
        y = spec.frequencies
        t = spec.times
        S = spec.spectrogram
        ax = axes_flat[i]
        im = ax.pcolormesh(t, y, S, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"Source {i+1}")
        ax.set_xlabel("Time (sec)")
        ax.set_ylabel("Frequency (Hz)")
        # add vertical markers relative to this plotted audio
        if mark_at_s is not None:
            for s in mark_at_s:
                ax.axvline(x=s, color='b')

    # hide any unused axes
    for j in range(n_sources, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    plt.show()

    # play/display combined buffered audio (same behavior as plot_clip)
    if play_audio:
        ipd.display(ipd.Audio(audio.samples, rate=audio.sample_rate, autoplay=True))


# ----------------------------------------------------------------------------------
# user_input_mixit
# ----------------------------------------------------------------------------------
def user_input_mixit(valid_choices=None, sources_column='mixit_sources', shift_column='shift'):
    """Prompt user for mixit separated sources (a comma-separated list of integers 1..8),
    then a numeric 'shift' (optional, press Enter to skip), then notes. Returns tuple
    (sources_str, shift_val, notes_str).
    """

    if valid_choices is None:
        valid_choices = [str(i) for i in range(1, 9)]

    while True:
        sources_raw = input(f"Enter sources as comma-separated integers 1-8 (e.g. 1,2,3) or press Enter to skip:")
        sources_raw = str(sources_raw).strip()
        if sources_raw == "":
            sources_str = ''
            break
        # validate format
        parts = [p.strip() for p in sources_raw.split(',') if p.strip()!='']
        if all(p in valid_choices for p in parts):
            # store as canonical comma-separated string '1,2,3'
            sources_str = ','.join(parts)
            break
        else:
            print('Invalid input. Use numbers 1..8 separated by commas.')
            continue

    # shift value (numeric) optional
    while True:
        shift_raw = input("Enter numeric shift value (or press Enter to skip): ").strip()
        if shift_raw == '':
            shift_val = np.NaN
            break
        try:
            # allow floats or ints
            if '.' in shift_raw:
                shift_val = float(shift_raw)
            else:
                shift_val = int(shift_raw)
            break
        except Exception:
            print('Invalid numeric input for shift. Try again or press Enter to skip.')
            continue

    notes = str(input('Enter any notes you would like to make or press enter to skip.\n'))

    proceed = input(f"Does this look right? Pressing 'r' to try again.\n").lower()
    if proceed == 'r':
        return user_input_mixit(valid_choices=valid_choices, sources_column=sources_column, shift_column=shift_column)

    return sources_str, shift_val, notes


# ----------------------------------------------------------------------------------
# Helpers for loading/saving (adapted from original)
# ----------------------------------------------------------------------------------
def save_annotations_file(annotations_df, scores_csv_path):
    """Saves annotations csv at [scores_csv_path] with '_mixit_annotations' suffix"""
    annotations_df.to_csv(f"{scores_csv_path.split('.')[0]}_mixit_annotations.csv")


def load_scores_df(scores_csv_path,
                   sources_column='mixit_sources',
                   shift_column='shift',
                   index_cols='relative_path',
                   notes_column='notes',
                   sort_by=None,
                   dry_run=False):
    """Load detection scores CSV data to be annotated (mixit version). If a previously saved
    mixit annotations CSV exists it will be loaded instead.
    Returns: (scores_df, annotation_csv_exists)
    """
    try:
        scores_df = pd.read_csv(f"{scores_csv_path.split('.')[0]}_mixit_annotations.csv")
        scores_df = scores_df.set_index(index_cols)
        annotation_csv_exists = True
    except Exception:
        scores_df = pd.read_csv(scores_csv_path)
        scores_df = scores_df.set_index(index_cols)
        # Create mixit-specific columns
        scores_df[sources_column] = np.NaN
        scores_df[shift_column] = np.NaN
        scores_df[notes_column] = np.NaN
        if not dry_run:
            save_annotations_file(scores_df, scores_csv_path)
        annotation_csv_exists = False

    if sort_by is not None:
        scores_df = scores_df.sort_values(sort_by)

    return scores_df, annotation_csv_exists


# ----------------------------------------------------------------------------------
# annotate_mixit
# ----------------------------------------------------------------------------------
def annotate_mixit(scores_file = "_scores.csv",
                   audio_dir = None,
                   sources_column = 'mixit_sources',
                   shift_column = 'shift',
                   notes_column = 'notes',
                   index_cols = ['relative_path'],
                   skip_cols = None,
                   n_positives = 1,
                   n_negatives = 100,
                   mark_at_s = None,
                   sort_by = None,
                   date_filter = [],
                   card_filter = [],
                   model = None,
                   n_sample = None,
                   dry_run = False,
                   buffer = None):
    """
    Annotate using mixit separation plotting + collect per-clip list of source indices and a numeric shift.
    Stores results in columns [sources_column] (string like '1,2,3'), [shift_column] (numeric) and [notes_column].
    """
    # locate csv
    if audio_dir:
        scores_csv_path = os.path.join(audio_dir, scores_file)
    else:
        scores_csv_path = scores_file

    scores_df, annotation_csv_exists = load_scores_df(scores_csv_path,
                                                     sources_column = sources_column,
                                                     shift_column = shift_column,
                                                     index_cols = index_cols,
                                                     notes_column = notes_column,
                                                     sort_by = sort_by,
                                                     dry_run = dry_run)

    # Skip filter column
    if date_filter or card_filter:
        scores_df['skip'] = (scores_df['date'].isin(date_filter)) | (scores_df['card'].isin(card_filter))
    else:
        scores_df['skip'] = False

    # rows not yet annotated (sources_column is NaN)
    valid_rows = scores_df[scores_df[sources_column].isna()]
    if n_sample is not None:
        valid_rows = valid_rows.sample(n_sample)

    n_clips = len(scores_df)
    n_clips_remaining = len(valid_rows)

    while len(valid_rows) > 0:
        row = valid_rows.iloc[0]
        idx = valid_rows.index[0]

        ipd.clear_output(wait = True)

        annotated_total = n_clips - n_clips_remaining
        print(f'{annotated_total} of {n_clips}')

        if row['skip']:
            scores_df.at[idx, sources_column] = 'not reviewed'
        else:
            print(f"Clip: {idx}")
            # call plot_clip_mixit: adapt to index_cols shape
            if len(index_cols) == 1:
                plot_clip_mixit(idx, mark_at_s = mark_at_s, buffer = buffer, model = model)
            elif len(index_cols) == 3:
                plot_clip_mixit(idx[0], idx[1], idx[2], mark_at_s = mark_at_s, buffer = buffer, model = model)
            else:
                raise Exception('index_cols must be either ["path_to_clip"] or ["path_to_audio", "clip_start_time", "clip_end_time"]')

            time.sleep(.1)
            sources_str, shift_val, notes = user_input_mixit()

            scores_df.loc[idx, sources_column] = sources_str
            scores_df.loc[idx, shift_column] = shift_val
            scores_df.loc[idx, notes_column] = notes

        if not dry_run:
            # drop helper 'skip' before saving
            save_df = scores_df.drop(['skip'], axis = 1)
            save_annotations_file(save_df, scores_csv_path)

        # update iterators
        n_clips_remaining = len(scores_df[scores_df[sources_column].isna()])
        valid_rows = scores_df[scores_df[sources_column].isna()]

    return scores_df
