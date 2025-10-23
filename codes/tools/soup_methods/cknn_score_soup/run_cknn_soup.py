#!/usr/bin/env python3
"""
Quick test script for CKNN Score-soup
"""

import sys
from pathlib import Path
from codes.cknn_soup_pipeline import run_cknn_soup_evaluation, setup_logger

def main():
    logger = setup_logger(verbose=True)
    
    # Example configuration for quick test
    config = {
        'dataset_name': 'shanghaitech',
        'uvadmode': 'merge', 
        'appae_variations': ['seed111', 'seed222', 'seed333'],  # Adjust based on your trained models
        'cleanse_thresholds': [75, 80, 85],  # Top 25%, 20%, 15% removal
        'k_values': [10, 20],  # Smaller set for quick test
        'feature_types': ['app'],
        'weight_method': 'keep_ratio',
        'output_dir': Path('results/cknn_soup_test')
    }
    
    logger.info("Running CKNN Score-soup quick test")
    logger.info(f"Config: {config}")
    
    try:
        results = run_cknn_soup_evaluation(**config, logger=logger)
        
        # Print summary
        print("\n" + "="*50)
        print("CKNN SCORE-SOUP RESULTS SUMMARY")
        print("="*50)
        
        print(f"\nDataset: {results['dataset_name']} ({results['uvadmode']})")
        print(f"AppAE variations: {results['appae_variations']}")
        print(f"Cleanse thresholds: {results['cleanse_thresholds']}")
        print(f"k values: {results['k_values']}")
        
        print(f"\nIndividual grader results: {len(results['grader_results'])}")
        for gr_result in results['grader_results'][:5]:  # Show first 5
            print(f"  {gr_result['variation_name']}: "
                  f"keep_ratio={gr_result['keep_ratio']:.3f}, "
                  f"n_train={gr_result['n_train']}, "
                  f"score_range=[{gr_result['scores_stats']['min']:.3f}, "
                  f"{gr_result['scores_stats']['max']:.3f}]")
        if len(results['grader_results']) > 5:
            print(f"  ... and {len(results['grader_results']) - 5} more")
        
        print(f"\nSoup results:")
        for feature_type, soup_result in results['soup_results'].items():
            print(f"  {feature_type}: {soup_result['n_graders']} graders combined")
            print(f"    Score range: [{soup_result['scores_stats']['min']:.3f}, "
                  f"{soup_result['scores_stats']['max']:.3f}]")
            print(f"    Score stats: mean={soup_result['scores_stats']['mean']:.3f}, "
                  f"std={soup_result['scores_stats']['std']:.3f}")
        
        print(f"\nResults saved to: {config['output_dir']}")
        print("\nTo evaluate AUROC, run:")
        print(f"python main2_evaluate.py --dataset_name {config['dataset_name']} "
              f"--mode {config['uvadmode']} --config configs/config_{config['dataset_name']}.yaml")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    main()