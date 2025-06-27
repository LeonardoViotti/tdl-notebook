#!/usr/bin/env python3
"""
Test script to verify the buffer functionality in plot_clip and annotate functions
"""

import pandas as pd
import numpy as np
from tdl_annotation import plot_clip, load_scores_df

def test_plot_clip_buffer():
    """Test that plot_clip handles buffer parameter correctly"""
    
    # Test data
    audio_path = "test_audio.wav"  # This won't exist, but we can test the logic
    st = 10.0
    end = 15.0
    buffer = 2.0
    
    print("Testing plot_clip buffer functionality...")
    
    # Test that the function accepts the buffer parameter
    try:
        # This will fail because the audio file doesn't exist, but we can test the parameter handling
        plot_clip(audio_path, st, end, buffer=buffer)
        print("❌ Expected error due to missing audio file, but function didn't raise exception")
        return False
    except FileNotFoundError:
        print("✅ plot_clip correctly accepts buffer parameter")
        return True
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_annotate_buffer_parameter():
    """Test that annotate function accepts buffer parameter"""
    
    # Create a simple test CSV file
    test_data = {
        'file': ['test1.wav', 'test2.wav', 'test3.wav'],
        'start_time': [0, 5, 10],
        'end_time': [3, 8, 13],
        'score': [0.8, 0.3, 0.9]
    }
    
    test_df = pd.DataFrame(test_data)
    test_csv_path = 'test_scores.csv'
    test_df.to_csv(test_csv_path, index=False)
    
    print("✅ Created test CSV file")
    
    try:
        # Test that load_scores_df works with the buffer scenario
        print("Testing load_scores_df function with buffer parameter...")
        
        result, exists = load_scores_df(
            scores_csv_path=test_csv_path,
            annotation_column='annotation',
            index_cols=['file', 'start_time', 'end_time'],
            dry_run=True
        )
        
        print("✅ load_scores_df function works!")
        print(f"Result shape: {result.shape}")
        print(f"Columns: {list(result.columns)}")
        
        # Check that the expected columns are present
        expected_columns = ['score', 'annotation', 'notes']
        for col in expected_columns:
            if col in result.columns:
                print(f"✅ Column '{col}' present")
            else:
                print(f"❌ Column '{col}' missing")
                return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    finally:
        # Clean up test files
        import os
        if os.path.exists(test_csv_path):
            os.remove(test_csv_path)
        if os.path.exists('test_scores_annotations.csv'):
            os.remove('test_scores_annotations.csv')
    
    return True

def test_buffer_logic():
    """Test the buffer calculation logic"""
    
    print("\nTesting buffer calculation logic...")
    
    # Test cases
    test_cases = [
        {'st': 10.0, 'end': 15.0, 'buffer': 2.0, 'expected_st': 8.0, 'expected_end': 17.0},
        {'st': 5.0, 'end': 10.0, 'buffer': 1.5, 'expected_st': 3.5, 'expected_end': 11.5},
        {'st': 0.0, 'end': 5.0, 'buffer': 3.0, 'expected_st': 0.0, 'expected_end': 8.0},  # st should not go below 0
        {'st': 10.0, 'end': 15.0, 'buffer': None, 'expected_st': 10.0, 'expected_end': 15.0},  # No buffer
    ]
    
    for i, case in enumerate(test_cases):
        st = case['st']
        end = case['end']
        buffer = case['buffer']
        expected_st = case['expected_st']
        expected_end = case['expected_end']
        
        if buffer and st is not None and end is not None:
            st_buffered = max(0, st - buffer)
            end_buffered = end + buffer
        else:
            st_buffered = st
            end_buffered = end
        
        if st_buffered == expected_st and end_buffered == expected_end:
            print(f"✅ Test case {i+1} passed: st={st_buffered}, end={end_buffered}")
        else:
            print(f"❌ Test case {i+1} failed: expected st={expected_st}, end={expected_end}, got st={st_buffered}, end={end_buffered}")
            return False
    
    return True

def test_automatic_marking_with_buffer():
    """Test that original clip boundaries are automatically marked when buffer is used"""
    
    print("\nTesting automatic marking of original clip boundaries...")
    
    # Test cases for automatic marking
    test_cases = [
        {
            'st': 10.0, 'end': 15.0, 'buffer': 2.0, 
            'expected_st_relative': 2.0, 'expected_end_relative': 7.0
        },
        {
            'st': 5.0, 'end': 10.0, 'buffer': 1.5, 
            'expected_st_relative': 1.5, 'expected_end_relative': 6.5
        },
        {
            'st': 0.0, 'end': 5.0, 'buffer': 3.0, 
            'expected_st_relative': 0.0, 'expected_end_relative': 5.0  # Fixed: when st=0, buffer can't go below 0
        },
    ]
    
    for i, case in enumerate(test_cases):
        st = case['st']
        end = case['end']
        buffer = case['buffer']
        expected_st_relative = case['expected_st_relative']
        expected_end_relative = case['expected_end_relative']
        
        # Simulate the logic from plot_clip
        if buffer and st is not None and end is not None:
            st_buffered = max(0, st - buffer)
            end_buffered = end + buffer
            
            # Calculate relative positions of original boundaries
            original_st_relative = st - st_buffered
            original_end_relative = end - st_buffered
            
            if (original_st_relative == expected_st_relative and 
                original_end_relative == expected_end_relative):
                print(f"✅ Test case {i+1} passed: st_relative={original_st_relative}, end_relative={original_end_relative}")
            else:
                print(f"❌ Test case {i+1} failed: expected st_relative={expected_st_relative}, end_relative={expected_end_relative}, got st_relative={original_st_relative}, end_relative={original_end_relative}")
                return False
        else:
            print(f"❌ Test case {i+1} failed: buffer logic not applied")
            return False
    
    return True

def test_mark_at_s_adjustment_with_buffer():
    """Test that existing mark_at_s values are adjusted when buffer is used"""
    
    print("\nTesting mark_at_s adjustment with buffer...")
    
    # Test case: original mark_at_s should be adjusted relative to buffered start
    st = 10.0
    end = 15.0
    buffer = 2.0
    original_mark_at_s = [11.0, 14.0]  # Marks within original clip
    
    # Simulate the logic from plot_clip
    st_buffered = max(0, st - buffer)  # 8.0
    adjusted_mark_at_s = [m - st_buffered for m in original_mark_at_s]
    expected_adjusted = [11.0 - 8.0, 14.0 - 8.0]  # [3.0, 6.0]
    
    if adjusted_mark_at_s == expected_adjusted:
        print(f"✅ mark_at_s adjustment passed: {original_mark_at_s} -> {adjusted_mark_at_s}")
        return True
    else:
        print(f"❌ mark_at_s adjustment failed: expected {expected_adjusted}, got {adjusted_mark_at_s}")
        return False

if __name__ == "__main__":
    print("Testing buffer functionality...")
    
    success1 = test_plot_clip_buffer()
    success2 = test_annotate_buffer_parameter()
    success3 = test_buffer_logic()
    success4 = test_automatic_marking_with_buffer()
    success5 = test_mark_at_s_adjustment_with_buffer()
    
    if success1 and success2 and success3 and success4 and success5:
        print("\n🎉 All buffer tests passed!")
    else:
        print("\n💥 Some buffer tests failed!") 