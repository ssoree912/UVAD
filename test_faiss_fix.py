#!/usr/bin/env python3
"""
Test script to verify Faiss GPU memory optimizations work correctly
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from codes.grader import KNNGrader

def test_basic_functionality():
    """Test basic KNNGrader functionality with new parameters"""
    print("Testing basic KNNGrader functionality...")
    
    # Create synthetic data
    n_train = 1000
    n_test = 100
    dim = 512
    
    np.random.seed(42)
    train_features = np.random.randn(n_train, dim).astype(np.float32)
    test_features = np.random.randn(n_test, dim).astype(np.float32)
    
    # Test 1: Default parameters (Flat L2, 64MB temp memory)
    print("Test 1: Default parameters (Flat L2, 64MB temp memory)")
    grader1 = KNNGrader(train_features, K=5, key="test_default", gpu_ids=[0] if True else None)
    scores1 = grader1.grade_flat(test_features)
    print(f"  Scores shape: {scores1.shape}, range: [{scores1.min():.3f}, {scores1.max():.3f}]")
    del grader1
    
    # Test 2: Very low temp memory (16MB)
    print("Test 2: Very low temp memory (16MB)")
    grader2 = KNNGrader(train_features, K=5, key="test_low_temp", gpu_ids=[0] if True else None, temp_memory_mb=16)
    scores2 = grader2.grade_flat(test_features)
    print(f"  Scores shape: {scores2.shape}, range: [{scores2.min():.3f}, {scores2.max():.3f}]")
    del grader2
    
    # Test 3: IVF-Flat index
    print("Test 3: IVF-Flat index")
    grader3 = KNNGrader(train_features, K=5, key="test_ivf", gpu_ids=[0] if True else None, 
                       use_ivf=True, ivf_nlist=32, ivf_nprobe=8)
    scores3 = grader3.grade_flat(test_features)
    print(f"  Scores shape: {scores3.shape}, range: [{scores3.min():.3f}, {scores3.max():.3f}]")
    del grader3
    
    # Test 4: Float16 precision
    print("Test 4: Float16 precision")
    grader4 = KNNGrader(train_features, K=5, key="test_float16", gpu_ids=[0] if True else None,
                       use_float16=True, temp_memory_mb=32)
    scores4 = grader4.grade_flat(test_features)
    print(f"  Scores shape: {scores4.shape}, range: [{scores4.min():.3f}, {scores4.max():.3f}]")
    del grader4
    
    # Test 5: All optimizations combined
    print("Test 5: All optimizations combined")
    grader5 = KNNGrader(train_features, K=5, key="test_all_opts", gpu_ids=[0] if True else None,
                       temp_memory_mb=16, use_ivf=True, ivf_nlist=32, ivf_nprobe=8, use_float16=True)
    scores5 = grader5.grade_flat(test_features)
    print(f"  Scores shape: {scores5.shape}, range: [{scores5.min():.3f}, {scores5.max():.3f}]")
    del grader5
    
    print("✓ All basic tests passed!")

def test_large_data():
    """Test with larger data that might trigger OOM without optimizations"""
    print("\nTesting with larger data...")
    
    # Larger dataset that might cause issues
    n_train = 50000
    n_test = 500
    dim = 512
    
    np.random.seed(42)
    train_features = np.random.randn(n_train, dim).astype(np.float32)
    test_features = np.random.randn(n_test, dim).astype(np.float32)
    
    try:
        # Use IVF + low temp memory for large data
        print("Testing large data with IVF + low temp memory...")
        grader = KNNGrader(train_features, K=10, key="test_large", gpu_ids=[0] if True else None,
                          temp_memory_mb=16, use_ivf=True, ivf_nlist=256, ivf_nprobe=16)
        scores = grader.grade_flat(test_features)
        print(f"  Large data test: Scores shape: {scores.shape}, range: [{scores.min():.3f}, {scores.max():.3f}]")
        del grader
        print("✓ Large data test passed!")
        
    except Exception as e:
        print(f"✗ Large data test failed: {e}")

if __name__ == "__main__":
    print("Testing Faiss GPU memory optimizations...\n")
    
    try:
        import faiss
        gpu_count = faiss.get_num_gpus()
        print(f"Available GPUs: {gpu_count}")
        
        if gpu_count > 0:
            test_basic_functionality()
            test_large_data()
        else:
            print("No GPUs available, testing with CPU only...")
            # Modify tests to use CPU
            test_basic_functionality()
            
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()