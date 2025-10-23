#!/usr/bin/env python3
"""
Test script for the complete CKNN pipeline with App+Mot support
"""

import subprocess
import sys
from pathlib import Path

def test_app_only():
    """Test App signal only"""
    print("=== Testing App signal only ===")
    cmd = [
        sys.executable, "-m", "codes.tools.soup_methods.cknn_score_soup.cknn_soup_pipeline",
        "--dataset_name", "shanghaitech",
        "--uvadmode", "merge", 
        "--appae_variations", "seed111", "seed222", "seed333",
        "--cleanse_thresholds", "75", "80", "85",
        "--k_values", "10", "20",
        "--feature_types", "app",
        "--max_train_samples", "10000",  # Smaller for testing
        "--coreset_method", "random",
        "--gpu_list", "0", "1",
        "--output_dir", "results/cknn_test/app_only",
        "--verbose"
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0

def test_mot_only():
    """Test Mot signal only"""
    print("=== Testing Mot signal only ===")
    cmd = [
        sys.executable, "-m", "codes.tools.soup_methods.cknn_score_soup.cknn_soup_pipeline",
        "--dataset_name", "shanghaitech",
        "--uvadmode", "merge",
        "--appae_variations", "seed111",  # Will be ignored for MOT
        "--cleanse_thresholds", "75", "80", "85", 
        "--k_values", "10", "20",
        "--feature_types", "mot",
        "--max_train_samples", "10000",  # Smaller for testing
        "--coreset_method", "random",
        "--gpu_list", "0", "1",
        "--output_dir", "results/cknn_test/mot_only",
        "--verbose"
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0

def test_app_mot_combined():
    """Test App+Mot combined (full CKNN)"""
    print("=== Testing App+Mot combined (full CKNN) ===")
    cmd = [
        sys.executable, "-m", "codes.tools.soup_methods.cknn_score_soup.cknn_soup_pipeline",
        "--dataset_name", "shanghaitech",
        "--uvadmode", "merge",
        "--appae_variations", "seed111", "seed222", "seed333",
        "--cleanse_thresholds", "75", "80", "85",
        "--k_values", "10", "20",
        "--feature_types", "app", "mot",  # Both signals
        "--max_train_samples", "10000",  # Smaller for testing
        "--coreset_method", "random", 
        "--gpu_list", "0", "1",
        "--output_dir", "results/cknn_test/app_mot_combined",
        "--verbose"
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0

def main():
    """Run all tests"""
    print("Testing complete CKNN pipeline...")
    
    # Create output directory
    Path("results/cknn_test").mkdir(parents=True, exist_ok=True)
    
    tests = [
        ("App only", test_app_only),
        ("Mot only", test_mot_only), 
        ("App+Mot combined", test_app_mot_combined)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running test: {test_name}")
        print(f"{'='*50}")
        
        try:
            success = test_func()
            results.append((test_name, success))
            print(f"✅ {test_name}: {'PASSED' if success else 'FAILED'}")
        except Exception as e:
            results.append((test_name, False))
            print(f"❌ {test_name}: FAILED with exception: {e}")
    
    # Print summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    # Overall result
    all_passed = all(success for _, success in results)
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())