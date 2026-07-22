

from .lightning_module import CaloINNLightningModule
from .classifier import MLP
import torch 


class ResamplingBaseDistributionCaloINN(CaloINNLightningModule):
    """CaloINN with resampling base distribution.

    This is a subclass of CaloINNLightningModule that overrides the base distribution
    to be a resampling distribution instead of a standard Gaussian. The resampling
    distribution is defined by a set of samples drawn from the target distribution.
    """

    def __init__(
        self,
        setup_data_sample_path=None,
        enable_diagnostics=False,
        actnorm_calibration_samples=1024,
        cinn_params={},
        width_noise=1e-7,
        optimizer_params=None,
        scheduler_params=None,
        # Phase 2: accept preprocessed data arrays directly, bypassing file read
        init_data_x=None,
        init_data_c=None,
        init_layer_boundaries=None,
        init_num_train_samples=None,
        init_from_legacy_train_split=False,
        # max_samples: randomly subsample init data (matches legacy behavior)
        max_samples=None,

        clf_hidden_dim=256,
        clf_num_layers=4,
        clf_layer_norm=True,
        clf_dropout=0.0,
        clf_activation="relu",
        truncation_parameter: int=100,
        **kwargs,
    ):
        super().__init__(
            setup_data_sample_path=None,
            enable_diagnostics=False,
            actnorm_calibration_samples=1024,
            cinn_params={},
            width_noise=1e-7,
            optimizer_params=None,
            scheduler_params=None,
            # Phase 2: accept preprocessed data arrays directly, bypassing file read
            init_data_x=None,
            init_data_c=None,
            init_layer_boundaries=None,
            init_num_train_samples=None,
            init_from_legacy_train_split=False,
            # max_samples: randomly subsample init data (matches legacy behavior)
            max_samples=None, 
            **kwargs
        )

        self.acceptance_fn = MLP(
            hidden_dim=clf_hidden_dim,
            num_layers=clf_num_layers,
            layer_norm=clf_layer_norm,
            dropout=clf_dropout,
            activation=clf_activation,
            output_dim=1,
        )

        self.truncation_parameter = truncation_parameter
    


