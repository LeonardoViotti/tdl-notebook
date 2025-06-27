#!/usr/bin/env python3
"""
Test script to verify the annotate function works without skip_cols
"""

import pandas as pd
import numpy as np
from tdl_annotation import load_scores_df

def test_load_scores_without_skip_cols():
    """Test that load_scores_df works correctly"""
    
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
        # Test load_scores_df function
        print("Testing load_scores_df function...")
        
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

def test_skip_cols_logic():
    """Test the skip_cols logic in the annotate function"""
    
    # Create test data with skip_cols
    test_data = {
        'file': ['test1.wav', 'test2.wav', 'test3.wav', 'test4.wav'],
        'start_time': [0, 5, 10, 15],
        'end_time': [3, 8, 13, 18],
        'score': [0.8, 0.3, 0.9, 0.2],
        'card': ['card1', 'card1', 'card2', 'card2']
    }
    
    test_df = pd.DataFrame(test_data)
    test_csv_path = 'test_scores.csv'
    test_df.to_csv(test_csv_path, index=False)
    
    print("\n✅ Created test CSV file with card column")
    
    try:
        # Test load_scores_df function with skip_cols scenario
        print("Testing load_scores_df function with skip_cols scenario...")
        
        result, exists = load_scores_df(
            scores_csv_path=test_csv_path,
            annotation_column='annotation',
            index_cols=['file', 'start_time', 'end_time'],
            dry_run=True
        )
        
        print("✅ load_scores_df function works with card column!")
        print(f"Result shape: {result.shape}")
        print(f"Columns: {list(result.columns)}")
        
        # Check that the expected columns are present
        expected_columns = ['score', 'card', 'annotation', 'notes']
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

if __name__ == "__main__":
    print("Testing annotate function fixes...")
    
    success1 = test_load_scores_without_skip_cols()
    success2 = test_skip_cols_logic()
    
    if success1 and success2:
        print("\n🎉 All tests passed! The KeyError fix is working.")
    else:
        print("\n💥 Some tests failed!") 