#!/usr/bin/env python3
"""
Diagnostic: Compare model behavior in untrained vs trained state
"""

import torch
import math
import sys
sys.path.insert(0, './src')

from model import CINN


def create_realistic_data(batch_size=32, num_features=373, seed=42):
    """Create data similar to real calorimeter data"""
    torch.manual_seed(seed)
    
    # Actual calorimeter data characteristics:
    # - Mostly small values (energy deposits)
    # - Some exact zeros (cells with no energy)
    # - Sparse structure
    
    x = torch.rand(batch_size, num_features)
    # Make it mostly sparse with values in [0, 1]
    x[x < 0.7] = 0  # 70% zeros
    x[x >= 0.7] = torch.rand_like(x[x >= 0.7]) * 10 + 0.01
    
    # Condition (energy)
    c = torch.ones(batch_size, 1) * 100
    
    return x, c


def test_log_prob_behavior():
    """Test log_prob computation with different data and model states"""
    print("\n" + "="*70)
    print("TEST: log_prob behavior in untrained model")
    print("="*70)
    
    # Create training data
    train_x, train_c = create_realistic_data(1000, 373)
    
    params = {
        "data_path": "dummy",
        "n_layers": 4,
        "hidden_size": 128,
        "internal_size": 128,
        "n_blocks": 4,
        "width_noise": 1e-7,
        "eps": 1e-10,
        "bayesian": False,
    }
    
    print("\nCreating untrained CINN model...")
    model = CINN(params, train_x, train_c)
    model.eval()
    print(f"Created with {sum(p.numel() for p in model.parameters())} trainable parameters")
    
    # Test different data patterns
    test_patterns = [
        ("Training data (mixed, sparse)", train_x[:32], train_c[:32]),
        ("All zeros", torch.zeros(32, 373), torch.ones(32, 1) * 100),
        ("All ones", torch.ones(32, 373), torch.ones(32, 1) * 100),
        ("Very small uniform values", torch.ones(32, 373) * 1e-8, torch.ones(32, 1) * 100),
        ("Large values", torch.ones(32, 373) * 1000, torch.ones(32, 1) * 100),
        ("Mixed: 90% zeros, 10% normal", 
         torch.cat([torch.zeros(29, 373), train_x[:3]], dim=0), 
         torch.ones(32, 1) * 100),
    ]
    
    for name, test_x, test_c in test_patterns:
        print(f"\n--- {name} ---")
        zeros_count = (test_x==0).sum().item()
        x_min = test_x[test_x!=0].min().item() if (test_x!=0).any() else 0.
        x_max = test_x.max().item()
        c_min = test_c.min().item()
        c_max = test_c.max().item()
        print(f"Input x: zeros={zeros_count}/{test_x.numel()}, "
              f"min={x_min:.6e}, max={x_max:.6e}")
        print(f"Input c: min={c_min:.6e}, max={c_max:.6e}")
        
        try:  
            # Compute log_prob
            with torch.no_grad():
                log_probs = model.log_prob(test_x, test_c)
            
            print(f"log_probs: "
                  f"finite={torch.isfinite(log_probs).sum()}/{log_probs.shape[0]}, "
                  f"nan={torch.isnan(log_probs).sum()}, "
                  f"inf={torch.isinf(log_probs).sum()}")
            
            if torch.isfinite(log_probs).any():
                finite_vals = log_probs[torch.isfinite(log_probs)]
                print(f"  Finite range: min={finite_vals.min():.6e}, max={finite_vals.max():.6e}")
            
            # If there are NaN/inf values, investigate forward pass
            if (~torch.isfinite(log_probs)).any():
                print(f"\n  Investigating non-finite values...")
                with torch.no_grad():
                    z, log_jac_det = model.forward(test_x, test_c, rev=False)
                print(f"    z: finite={torch.isfinite(z).sum()}/{z.numel()}, "
                      f"nan={torch.isnan(z).sum()}, inf={torch.isinf(z).sum()}")
                print(f"    z range: min={z[torch.isfinite(z)].min():.6e}, max={z[torch.isfinite(z)].max():.6e}")
                print(f"    log_jac_det: finite={torch.isfinite(log_jac_det).sum()}/{log_jac_det.shape[0]}, "
                      f"nan={torch.isnan(log_jac_det).sum()}, inf={torch.isinf(log_jac_det).sum()}")
                print(f"    log_jac_det range: min={log_jac_det[torch.isfinite(log_jac_det)].min():.6e}, "
                      f"max={log_jac_det[torch.isfinite(log_jac_det)].max():.6e}")
                
                # Detailed log_prob computation
                z_sq_sum = torch.sum(z**2, dim=1)
                term1 = -0.5 * z_sq_sum
                term2 = log_jac_det
                term3 = -z.shape[1]/2 * math.log(2*math.pi)
                print(f"    -0.5*sum(z²): min={term1.min():.6e}, max={term1.max():.6e}")
                print(f"    log_jac_det: min={term2.min():.6e}, max={term2.max():.6e}")
                print(f"    constant: {term3:.6e}")
                
        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()


def simulate_training_steps():
    """Simulate a few training steps and see how model behavior changes"""
    print("\n" + "="*70)
    print("TEST: Model behavior change through training steps")
    print("="*70)
    
    # Create training data
    train_x, train_c = create_realistic_data(100, 373)
    test_x, test_c = create_realistic_data(32, 373, seed=123)
    
    params = {
        "data_path": "dummy",
        "n_layers": 2,  # Smaller for faster testing
        "hidden_size": 64,
        "internal_size": 64,
        "n_blocks": 2,
        "width_noise": 1e-7,
        "eps": 1e-10,
        "bayesian": False,
        "lr": 0.001,
    }
    
    print("\nCreating model...")
    model = CINN(params, train_x, train_c)
    
    # Simple training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("\nRunning training steps...")
    for step in range(5):
        model.train()
        batch_x = train_x[:32]
        batch_c = train_c[:32]
        
        # Forward pass
        log_probs = model.log_prob(batch_x, batch_c)
        loss = -torch.mean(log_probs)
        
        # Check if loss is finite
        loss_finite = torch.isfinite(loss).item()
        log_prob_finite_count = torch.isfinite(log_probs).sum().item()
        
        print(f"\nStep {step}:")
        print(f"  Training loss: {loss.item():.6e}, finite={loss_finite}")
        print(f"  log_probs: {log_prob_finite_count}/{log_probs.shape[0]} finite")
        
        if loss_finite:
            # Update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"  Updated model weights")
        else:
            print(f"  SKIPPED weight update (non-finite loss)")
        
        # Test on test data
        model.eval()
        with torch.no_grad():
            test_log_probs = model.log_prob(test_x, test_c)
        test_finite = torch.isfinite(test_log_probs).sum().item()
        print(f"  Test log_probs: {test_finite}/{test_log_probs.shape[0]} finite")


if __name__ == "__main__":
    print("\n[SIMPLIFIED DIAGNOSTIC]")
    print("Testing log_prob computation with untrained CINN\n")
    
    test_log_prob_behavior()
    simulate_training_steps()
    
    print("\n" + "="*70)
    print("Diagnostic complete")
    print("="*70)
