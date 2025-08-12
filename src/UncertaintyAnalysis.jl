module UncertaintyAnalysis

using Statistics
using StatsBase
using PyPlot
using ImageQualityIndexes
using Distributions
using JLD2

export UncertaintyAnalysisConfig, run_uncertainty_analysis

# Configuration structure
struct UncertaintyAnalysisConfig
    grid_spacing::Tuple{Float64, Float64}
    confidence_levels::Vector{Float64}
    calibration_bins::Int
    zscore_threshold::Float64
    vmax_velocity::Float64
    vmax_error::Float64
    vmax_uncertainty::Float64
    output_format::String
    dpi::Int
    verbose::Bool
end

# Include submodules
include("plotting.jl")
include("metrics.jl")
include("calibration.jl")
include("coverage.jl")
include("utils.jl")

"""
    run_uncertainty_analysis(ground_truth, posterior_samples, output_dir, plots_to_generate, config; posterior_mean=nothing)

Main function to run uncertainty analysis pipeline.

# Arguments
- `ground_truth`: 2D or 3D array with ground truth data
- `posterior_samples`: 4D array (nx, nz, 1, n_samples) with posterior samples
- `output_dir`: Output directory path
- `plots_to_generate`: Vector of plot names to generate
- `config`: UncertaintyAnalysisConfig object
- `posterior_mean`: Optional precomputed posterior mean

# Returns
- Dictionary with computed metrics and results
"""
function run_uncertainty_analysis(
    ground_truth,
    posterior_samples,
    output_dir::String,
    plots_to_generate::Vector{String},
    config::UncertaintyAnalysisConfig;
    posterior_mean=nothing
)
    results = Dict{String, Any}()
    
    # Ensure ground truth is 3D for consistency
    if ndims(ground_truth) == 2
        ground_truth = reshape(ground_truth, size(ground_truth)..., 1)
    end
    gt_2d = ground_truth[:, :, 1]
    
    # Compute posterior statistics
    if posterior_mean === nothing
        posterior_mean = mean(posterior_samples, dims=4)[:, :, 1, 1]
    end
    posterior_std = std(posterior_samples, dims=4)[:, :, 1, 1]
    
    config.verbose && println("Computing basic metrics...")
    
    # Compute basic metrics
    results["ssim"] = assess_ssim(posterior_mean, gt_2d)
    results["rmse"] = sqrt(mean((posterior_mean - gt_2d).^2))
    results["mean_uncertainty"] = mean(posterior_std)
    results["max_uncertainty"] = maximum(posterior_std)
    
    # Generate requested plots
    for plot_type in plots_to_generate
        config.verbose && println("Generating $plot_type plot...")
        
        if plot_type == "posterior_mean"
            plot_posterior_mean(posterior_mean, gt_2d, output_dir, config, results["ssim"])
            
        elseif plot_type == "error"
            plot_error_map(posterior_mean, gt_2d, output_dir, config, results["rmse"])
            
        elseif plot_type == "rmse"
            plot_rmse_map(posterior_samples, gt_2d, output_dir, config)
            
        elseif plot_type == "uncertainty" || plot_type == "std"
            plot_uncertainty_map(posterior_std, output_dir, config, results["mean_uncertainty"])
            
        elseif plot_type == "calibration"
            config.verbose && println("  Computing calibration metrics...")
            uce, errors_binned, uncert_binned = compute_calibration_metrics(
                posterior_samples, gt_2d, config.calibration_bins
            )
            results["uce"] = uce
            results["calibration_errors"] = errors_binned
            results["calibration_uncertainty"] = uncert_binned
            plot_calibration_curve(uce, uncert_binned, errors_binned, output_dir, config)
            
        elseif plot_type == "zscore"
            config.verbose && println("  Computing z-score analysis...")
            zscore_map, outlier_percentage = compute_zscore_analysis(
                posterior_samples, gt_2d, config.zscore_threshold
            )
            results["zscore_map"] = zscore_map
            results["outlier_percentage"] = outlier_percentage
            plot_zscore_map(zscore_map, outlier_percentage, output_dir, config)
            
        elseif plot_type == "coverage"
            config.verbose && println("  Computing coverage analysis...")
            coverage_stats = compute_coverage_statistics(
                posterior_samples, gt_2d, config.confidence_levels
            )
            results["coverage_stats"] = coverage_stats
            plot_coverage_analysis(coverage_stats, config.confidence_levels, output_dir, config)
            
        else
            @warn "Unknown plot type: $plot_type"
        end
    end
    
    return results
end

end # module