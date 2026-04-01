#!/usr/bin/env python3
"""
Deep diagnostic: trace through each CINN module to find where NaN appears
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


def trace_through_modules(model, x, c, verbose=True):
    """Trace through each CINN module and report statistics"""
    print("\nTracing through CINN modules:\n")
    print(f"Input x: shape={x.shape}, zeros={int((x==0).sum())}/{x.numel()}, "
          f"min={x[x!=0].min() if (x!=0).any() else 0:.6e}, "
          f"max={x.max():.6e}")
    print(f"Input c: shape={c.shape}, min={c.min():.6e}, max={c.max():.6e}\n")
    
    # Access model's GraphINN
    if hasattr(model, 'model'):
        inn = model.model
    else:
        inn = model
    
    # Start forward pass through modules
    z = x
    condition = c
    total_jac = torch.zeros(z.shape[0], device=z.device)
    
    if not hasattr(inn, 'module_list'):
        print(f"ERROR: Model does not have module_list. Attributes: {dir(inn)}")
        return
    
    for i, module in enumerate(inn.module_list):
        module_name = module.__class__.__name__
        print(f"[Module {i}] {module_name}:")
        
        try:
            if hasattr(module, 'dims_c') and hasattr(module, 'conditions_required'):
                # Transformation that uses conditions
                if module_name == 'MixedTransformation':
                    print(f"  Input z: shape={z.shape}, zeros={(z==0).sum()}, "
                          f"min={z[z!=0].min() if (z!=0).any() else 'all_zero':.6e}, "
                          f"max={z.max():.6e}")
                    # Check for problematic inputs to log operations
                    if (z <= 0).any():
                        problem_ratio = (z <= 0).sum() / z.numel()
                        print(f"    WARNING: {problem_ratio*100:.1f}% values <= 0 (problematic for log/logit)")
                    z, jac = module((z,), c, rev=False)
                    z = z[0]
                    total_jac = total_jac + jac
                elif hasattr(module, 'forward') and 'rev' in module.forward.__code__.co_varnames:
                    z, jac = module((z,), c, rev=False)
                    z = z[0]
                    total_jac = total_jac + jac
                else:
                    z = module(z, c)
            elif module_name == 'LogTransformation':
                print(f"  Input z: shape={z.shape}, zeros={(z==0).sum()}, "
                      f"min={z[z!=0].min() if (z!=0).any() else 'all_zero':.6e}, "
                      f"max={z.max():.6e}")
                # Check for problematic inputs to log
                if (z <= 0).any():
                    problem_ratio = (z <= 0).sum() / z.numel()
                    print(f"    WARNING: {problem_ratio*100:.1f}% values <= 0 (problematic for log)")
                z, jac = module((z,), rev=False)
                z = z[0]
                total_jac = total_jac + jac
            else:
                # Regular forward
                z = module(z)
            
            # Report output
            print(f"  Output z: shape={z.shape}, "
                  f"nan={(~torch.isfinite(z)).sum()}, inf={torch.isinf(z).sum()}, "
                  f"min={z[torch.isfinite(z)].min() if torch.isfinite(z).any() else 'all_inf/nan':.6e}, "
                  f"max={z[torch.isfinite(z)].max() if torch.isfinite(z).any() else 'all_inf/nan':.6e}")
            
            if (~torch.isfinite(z)).any():
                print(f"    ALERT: Non-finite values detected!")
                # Identify which samples have issues
                bad_samples = (~torch.isfinite(z)).any(dim=1)
                print(f"    Affected samples: {bad_samples.sum()}/{z.shape[0]}")
                
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            z_nan = (z != z) | (~torch.isfinite(z))
            print(f"  Current z: nan={z_nan.sum()}, inf={torch.isinf(z).sum()}")
    
    print(f"\nFinal z: shape={z.shape}, "
          f"nan={(~torch.isfinite(z)).sum()}, inf={torch.isinf(z).sum()}, "
          f"min={z[torch.isfinite(z)].min() if torch.isfinite(z).any() else 'all_inf/nan':.6e}, "
          f"max={z[torch.isfinite(z)].max() if torch.isfinite(z).any() else 'all_inf/nan':.6e}")
    print(f"Total jacobian: min={total_jac.min():.6e}, max={total_jac.max():.6e}, "
          f"nan={(~torch.isfinite(total_jac)).sum()}, inf={torch.isinf(total_jac).sum()}")
    
    return z, total_jac


def test_with_different_initializations():
    """Test with models in different states"""
    print("\n" + "="*70)
    print("TEST: CINN behavior with different data patterns")
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
    
    print("\nCreating CINN model...")
    model = CINN(params, train_x, train_c)
    print(f"Created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test different data patterns
    test_patterns = [
        ("Training data (normal)", train_x[:32], train_c[:32]),
        ("All zeros", torch.zeros(32, 373), torch.ones(32, 1) * 100),
        ("All ones", torch.ones(32, 373), torch.ones(32, 1) * 100),
        ("Very small values", torch.ones(32, 373) * 1e-8, torch.ones(32, 1) * 100),
        ("Mix: 90% zeros, 10% normal", 
         torch.cat([torch.zeros(29, 373), train_x[:3]], dim=0), 
         torch.ones(32, 1) * 100),
    ]
    
    for name, test_x, test_c in test_patterns:
        print(f"\n--- {name} ---")
        try:
            z, jac = trace_through_modules(model, test_x, test_c)
        except Exception as e:
            print(f"FATAL ERROR: {type(e).__name__}: {e}")


if __name__ == "__main__":
    print("\n[DEEP CINN DIAGNOSTIC]")
    print("Tracing through each CINN module\n")
    
    test_with_different_initializations()
    
    print("\n" + "="*70)
    print("Analysis complete")
    print("="*70)
