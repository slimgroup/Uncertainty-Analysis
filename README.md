# Uncertainty Analysis Toolkit

A comprehensive toolkit for uncertainty quantification and analysis of posterior samples from Bayesian inverse problems, particularly focused on image-based data.

## Overview

This repository provides functions for analyzing and visualizing uncertainty in posterior distributions, with emphasis on:
- Posterior sample analysis for image data
- Uncertainty quantification metrics
- Calibration analysis 
- Coverage assessment
- Statistical validation of Bayesian methods

## Core Functions

### 1. Posterior Summary Statistics
- **Posterior Mean Plot**: Compute and visualize the mean of posterior samples
- **Error Plot**: Absolute error between posterior mean and ground truth
- **Pixel-wise RMSE Plot**: Root mean square error computed pixel-wise across samples

### 2. Uncertainty Quantification
- **Uncertainty Calibration Plot**: Compute Uncertainty Calibration Error (UCE) to assess calibration quality
- **Z-score Plot**: Standardized residuals to assess distributional assumptions
- **Posterior Coverage Plot**: Empirical coverage assessment against theoretical confidence intervals

### 3. Input Requirements
All functions are designed to work with:
- **Posterior samples**: 4D array (height, width, channels, n_samples) 
- **Ground truth data**: 2D/3D array (height, width, [channels])
- **Image format**: Functions assume spatial data (e.g., velocity models, seismic images)

## Function Categories

### Visualization Functions
```julia
# Posterior analysis plots
plot_posterior_mean(posterior_samples, ground_truth; options...)
plot_error_map(posterior_samples, ground_truth; options...)
plot_rmse_map(posterior_samples, ground_truth; options...)

# Uncertainty assessment plots  
plot_uncertainty_calibration(posterior_samples, ground_truth; options...)
plot_zscore_analysis(posterior_samples, ground_truth; options...)
plot_coverage_analysis(posterior_samples, ground_truth; confidence_levels...)
```

### Metric Computation Functions
```julia
# Statistical metrics
compute_uce(posterior_samples, ground_truth)
compute_coverage_statistics(posterior_samples, ground_truth, confidence_levels)
compute_zscore_statistics(posterior_samples, ground_truth)
compute_pixelwise_rmse(posterior_samples, ground_truth)
```

### Utility Functions
```julia
# Data processing
validate_input_dimensions(posterior_samples, ground_truth)
compute_posterior_statistics(posterior_samples)
extract_confidence_intervals(posterior_samples, confidence_levels)
```

## Dependencies

- PyPlot.jl (for visualization)
- Statistics.jl (for statistical computations)
- StatsBase.jl (for advanced statistics)
- Images.jl (for image processing utilities)

## Usage Example

```julia
using UncertaintyAnalysis

# Load your data
posterior_samples = load_posterior_samples()  # Shape: (nx, nz, 1, n_samples)
ground_truth = load_ground_truth()           # Shape: (nx, nz)

# Generate comprehensive uncertainty analysis
plot_posterior_mean(posterior_samples, ground_truth, save_path="results/")
plot_error_map(posterior_samples, ground_truth, save_path="results/")
plot_uncertainty_calibration(posterior_samples, ground_truth, save_path="results/")
plot_coverage_analysis(posterior_samples, ground_truth, [0.68, 0.95], save_path="results/")

# Compute metrics
uce = compute_uce(posterior_samples, ground_truth)
coverage_stats = compute_coverage_statistics(posterior_samples, ground_truth, [0.68, 0.95])
```

## File Structure

```
Uncertainty-Analysis/
├── README.md
├── src/
│   ├── visualization/
│   │   ├── posterior_plots.jl
│   │   ├── error_plots.jl
│   │   ├── calibration_plots.jl
│   │   └── coverage_plots.jl
│   ├── metrics/
│   │   ├── calibration_metrics.jl
│   │   ├── coverage_metrics.jl
│   │   └── statistical_metrics.jl
│   └── utils/
│       ├── input_validation.jl
│       └── data_processing.jl
└── examples/
    └── uncertainty_analysis_example.jl
```

## Contributing

This toolkit is designed for robust uncertainty analysis in scientific computing applications. Functions should maintain consistency with image-based data formats and provide comprehensive uncertainty assessment capabilities.