#!/usr/bin/env python3
"""
Diagnostic script to trace NaN sources through the forward pass
without depending on external data files.
"""

import torch
import math
import sys
sys.path.insert(0, './src')

from model import CINN, LogTransformation


def create_synthetic_data(batch_size=32, num_features=373, num_conditions=1, seed=42):
    """Create synthetic training data"""
    torch.manual_seed(seed)
    
    # Create synthetic shower data (positive values, log-scale distribution)
    x = torch.exp(torch.randn(1000, num_features) * 2 - 3) * 100
    # Some zeros/very small values
    mask = torch.rand_like(x) > 0.9
    x[mask] = 1e-10
    
    # Condition (energy values)
    c = torch.exp(torch.randn(1000, num_conditions) * 2 + 5) * 10
    
    return x, c


def test_log_transformation():
    """Test LogTransformation directly"""
    print("\n" + "="*60)
    print("TEST 1: LogTransformation behavior")
    print("="*60)
    
    transform = LogTransformation(dims_in=[373])
    
    # Test with different input ranges
    test_cases = [
        ("normal positive values", torch.ones(10, 373) * 100),
        ("small values", torch.ones(10, 373) * 1e-6),
        ("very small values", torch.ones(10, 373) * 1e-12),
        ("mixed values", torch.cat([torch.ones(5, 373) * 100, torch.ones(5, 373) * 1e-12], dim=0)),
        ("with zeros", torch.cat([torch.ones(5, 373) * 100, torch.zeros(5, 373)], dim=0)),
    ]
    
    for name, x in test_cases:
        print(f"\n{name}:")
        print(f"  Input: min={x.min().item():.6e}, max={x.max().item():.6e}")
        try:
            z, jac = transform((x,), rev=False)
            z = z[0]
            print(f"  Output z: min={z.min().item():.6e}, max={z.max().item():.6e}, "
                  f"nan={torch.isnan(z).sum().item()}, inf={torch.isinf(z).sum().item()}")
            print(f"  Jacobian: min={jac.min().item():.6e}, max={jac.max().item():.6e}, "
                  f"nan={torch.isnan(jac).sum().item()}, inf={torch.isinf(jac).sum().item()}")
        except Exception as e:
            print(f"  ERROR: {e}")


def test_cinn_forward_pass():
    """Test CINN forward pass with synthetic data"""
    print("\n" + "="*60)
    print("TEST 2: CINN forward pass behavior")
    print("="*60)
    
    # Create synthetic data
    train_x, train_c = create_synthetic_data(1000, 373, 1)
    print(f"\nSynthetic data created:")
    print(f"  train_x: shape={train_x.shape}, min={train_x.min().item():.6e}, max={train_x.max().item():.6e}")
    print(f"  train_c: shape={train_c.shape}, min={train_c.min().item():.6e}, max={train_c.max().item():.6e}")
    
    # Create model parameters
    params = {
        "data_path": "dummy",  # Not used with provided tensors
        "n_layers": 4,
        "hidden_size": 128,
        "internal_size": 128,
        "n_blocks": 4,
        "width_noise": 1e-7,
        "eps": 1e-10,
        "bayesian": False,
    }
    
    # Create CINN
    print("\nInitializing CINN model...")
    model = CINN(params, train_x, train_c)
    print(f"CINN created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass with different input conditions
    test_batches = [
        ("batch from training data", train_x[:32], train_c[:32]),
        ("batch with very small values", torch.ones(32, 373) * 1e-10, torch.ones(32, 1) * 100),
        ("batch with mixed scales", 
         torch.cat([torch.ones(16, 373) * 100, torch.ones(16, 373) * 1e-10], dim=0),
         torch.cat([torch.ones(16, 1) * 100, torch.ones(16, 1) * 100], dim=0)),
    ]
    
    for name, x_batch, c_batch in test_batches:
        print(f"\n{name}:")
        print(f"  x: min={x_batch.min().item():.6e}, max={x_batch.max().item():.6e}")
        print(f"  c: min={c_batch.min().item():.6e}, max={c_batch.max().item():.6e}")
        
        try:
            # Forward pass
            z, log_jac_det = model.forward(x_batch, c_batch, rev=False)
            print(f"  z: min={z.min().item():.6e}, max={z.max().item():.6e}, "
                  f"nan={torch.isnan(z).sum().item()}, inf={torch.isinf(z).sum().item()}")
            print(f"  log_jac_det: min={log_jac_det.min().item():.6e}, max={log_jac_det.max().item():.6e}, "
                  f"nan={torch.isnan(log_jac_det).sum().item()}, inf={torch.isinf(log_jac_det).sum().item()}")
            
            # Compute log_prob
            z_sq_sum = torch.sum(z**2, dim=1)
            log_prob_term1 = -0.5 * z_sq_sum
            log_prob_term2 = log_jac_det
            log_prob_term3 = -z.shape[1]/2 * math.log(2*math.pi)
            log_prob = log_prob_term1 + log_prob_term2 + torch.tensor(log_prob_term3)
            
            print(f"  log_prob components:")
            print(f"    -0.5*sum(z²): min={log_prob_term1.min().item():.6e}, max={log_prob_term1.max().item():.6e}")
            print(f"    log_jac_det: min={log_prob_term2.min().item():.6e}, max={log_prob_term2.max().item():.6e}")
            print(f"    constant: {log_prob_term3:.6e}")
            print(f"  log_prob: min={log_prob.min().item():.6e}, max={log_prob.max().item():.6e}, "
                  f"nan={torch.isnan(log_prob).sum().item()}, inf={torch.isinf(log_prob).sum().item()}")
            
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")


if __name__ == "__main__":
    print("\n[NaN DIAGNOSTIC ANALYSIS]")
    print("Tracing numerical stability through forward pass\n")
    
    test_log_transformation()
    test_cinn_forward_pass()
    
    print("\n" + "="*60)
    print("Diagnostic complete")
    print("="*60)
