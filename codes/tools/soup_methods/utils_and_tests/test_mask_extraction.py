#!/usr/bin/env python3
"""
Test script to verify mask extraction and saving functionality.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.nn.utils import prune

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codes.cleanse import AppAE


def test_mask_extraction():
    """Test mask extraction from a pruned AppAE model."""
    print("Testing mask extraction functionality...")
    
    # Create a simple AppAE model
    model = AppAE()
    
    # Apply some pruning
    conv_modules = [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))]
    
    if conv_modules:
        print(f"Found {len(conv_modules)} conv modules")
        
        # Apply magnitude pruning to first few modules
        for i, module in enumerate(conv_modules[:3]):
            prune.l1_unstructured(module, name='weight', amount=0.2)
            print(f"Applied 20% magnitude pruning to module {i}")
        
        # Apply random pruning to some modules
        for i, module in enumerate(conv_modules[3:6]):
            prune.random_unstructured(module, name='weight', amount=0.1)
            print(f"Applied 10% random pruning to module {i+3}")
    
    # Test mask extraction
    masks = {}
    has_masks = False
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            # Check if this module has pruning masks
            if hasattr(module, 'weight_mask'):
                masks[f'{name}.weight'] = module.weight_mask.detach().cpu().clone()
                has_masks = True
                print(f"Found weight mask for {name}: shape={module.weight_mask.shape}, sparsity={1 - module.weight_mask.float().mean():.3f}")
            
            if hasattr(module, 'bias_mask') and module.bias is not None:
                masks[f'{name}.bias'] = module.bias_mask.detach().cpu().clone()
                has_masks = True
                print(f"Found bias mask for {name}")
    
    if has_masks:
        print(f"\nExtracted {len(masks)} masks total")
        
        # Test saving and loading
        test_path = "test_masks.pt"
        torch.save(masks, test_path)
        print(f"Saved masks to {test_path}")
        
        # Load and verify
        loaded_masks = torch.load(test_path, map_location='cpu')
        print(f"Loaded {len(loaded_masks)} masks")
        
        # Verify mask content
        for name, mask in loaded_masks.items():
            original_mask = masks[name]
            if torch.equal(mask, original_mask):
                print(f"✓ Mask {name} verified")
            else:
                print(f"✗ Mask {name} mismatch!")
        
        # Clean up
        Path(test_path).unlink()
        print(f"Cleaned up {test_path}")
    else:
        print("No masks found - pruning may not have been applied correctly")
    
    print("Test completed!")


def test_mask_loading_compatibility():
    """Test loading masks in the enhanced utils format."""
    print("\nTesting mask loading compatibility...")
    
    # Create dummy state dicts and checkpoint paths
    dummy_state = {"conv1.weight": torch.randn(32, 1, 3, 3)}
    dummy_ckpt_path = "dummy_checkpoint.pkl"
    
    # Create a dummy mask file
    dummy_masks = {
        "conv1.weight": torch.ones(32, 1, 3, 3) * 0.8  # 80% weights active
    }
    dummy_masks["conv1.weight"][0, 0, :, :] = 0  # Prune some weights
    
    mask_path = dummy_ckpt_path + ".mask"
    torch.save(dummy_masks, mask_path)
    print(f"Created dummy mask file: {mask_path}")
    
    # Test loading with enhanced utils
    try:
        from codes.tools.enhanced_appae_fisher_utils import load_masks
        loaded_masks = load_masks([dummy_state], [dummy_ckpt_path])
        
        if loaded_masks and loaded_masks[0] is not None:
            print("✓ Successfully loaded masks with enhanced utils")
            mask = loaded_masks[0]["conv1.weight"]
            print(f"  Mask shape: {mask.shape}")
            print(f"  Mask dtype: {mask.dtype}")
            print(f"  Active weights: {mask.sum()}/{mask.numel()} ({mask.float().mean():.3f})")
        else:
            print("✗ Failed to load masks")
    except ImportError as e:
        print(f"✗ Could not import enhanced utils: {e}")
    except Exception as e:
        print(f"✗ Error loading masks: {e}")
    
    # Clean up
    Path(mask_path).unlink()
    print(f"Cleaned up {mask_path}")


if __name__ == "__main__":
    test_mask_extraction()
    test_mask_loading_compatibility()