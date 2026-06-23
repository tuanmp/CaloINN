"""MCMC density ratio reweighting for CaloINN.

Provides classifier-based density ratio estimation and Independent
Metropolis-Hastings sampling to correct CINN-generated showers toward
the true Geant4 distribution.

Main classes
------------
- ``IMHSampler`` : Runs IMH chains with classifier-based acceptance.
- ``ClassifierWrapper`` : Loads trained MLP classifier checkpoints.
- ``TemperatureCalibrator`` : Temperature scaling for calibrated density ratios.
- ``cinn_sample_to_classifier_input`` : Converts CINN internal representation
  to the format expected by the classifier.
"""

from src.mcmc.calibration import (
    TemperatureCalibrator,
    calibrate_platt,
    compare_calibration_methods,
    expected_calibration_error,
)
from src.mcmc.classifier import ClassifierWrapper, MLP

__all__ = [
    "ClassifierWrapper",
    "MLP",
    "TemperatureCalibrator",
    "calibrate_platt",
    "compare_calibration_methods",
    "expected_calibration_error",
]
