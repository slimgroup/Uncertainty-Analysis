#!/usr/bin/env julia

"""
Example usage of the Uncertainty Analysis CLI Tool

This script demonstrates how to use the uncertainty analysis tool
both as a command-line interface and programmatically.
"""

using Pkg
Pkg.activate(".")

# Example 1: Command line usage
println("="^60)
println("EXAMPLE 1: Command Line Usage")
println("="^60)

println("""
# Basic usage with individual files:
julia uncertainty_analysis.jl \\
    --ground-truth data/velocity_gt.jld2 \\
    --posterior-samples data/velocity_posterior.jld2 \\
    --output ./results/ \\
    --plots posterior_mean,error,calibration

# Using configuration file:
julia uncertainty_analysis.jl --config example_config.toml

# Specify parameters:
julia uncertainty_analysis.jl \\
    --ground-truth data/gt.jld2 \\
    --posterior-samples data/post.jld2 \\
    --grid-spacing "6.25,6.25" \\
    --confidence-levels "0.68,0.95,0.99" \\
    --calibration-bins 64 \\
    --vmax-velocity 4.8 \\
    --output ./seismic_uncertainty/
""")

# Example 2: Programmatic usage
println("\n" * "="^60)
println("EXAMPLE 2: Programmatic Usage")
println("="^60)

include("src/UncertaintyAnalysis.jl")
using .UncertaintyAnalysis

# Create synthetic data for demonstration
function create_synthetic_data()
    """Create synthetic velocity model and posterior samples for testing"""
    nx, nz = 128, 64
    n_samples = 32
    
    # Create synthetic ground truth (layered velocity model)
    ground_truth = zeros(Float32, nx, nz)
    for i in 1:nx, j in 1:nz
        depth_factor = j / nz
        ground_truth[i, j] = 1.5 + 2.5 * depth_factor + 0.1 * sin(i/10) * cos(j/8)
    end
    
    # Create synthetic posterior samples with realistic uncertainty structure
    posterior_samples = zeros(Float32, nx, nz, 1, n_samples)
    for s in 1:n_samples
        for i in 1:nx, j in 1:nz
            # Base velocity from ground truth
            base_vel = ground_truth[i, j]
            
            # Add correlated noise (higher uncertainty in deeper layers)
            noise_level = 0.05 + 0.1 * (j / nz)  # Increasing uncertainty with depth
            noise = noise_level * randn()
            
            # Add some spatial correlation
            if i > 1 && j > 1
                spatial_corr = 0.3 * (posterior_samples[i-1, j, 1, s] + posterior_samples[i, j-1, 1, s]) / 2
                noise += 0.2 * (spatial_corr - base_vel)
            end
            
            posterior_samples[i, j, 1, s] = base_vel + noise
        end
    end
    
    return ground_truth, posterior_samples
end

println("Creating synthetic data...")
ground_truth, posterior_samples = create_synthetic_data()

println("Ground truth shape: $(size(ground_truth))")
println("Posterior samples shape: $(size(posterior_samples))")

# Create analysis configuration
config = UncertaintyAnalysisConfig(
    grid_spacing = (6.25, 6.25),
    confidence_levels = [0.68, 0.95],
    calibration_bins = 32,
    zscore_threshold = 2.0,
    vmax_velocity = 4.5,
    vmax_error = 0.4,
    vmax_uncertainty = 0.3,
    output_format = "png",
    dpi = 200,
    verbose = true
)

# Run analysis
output_dir = "./example_results/"
plots_to_generate = ["posterior_mean", "error", "uncertainty", "calibration", "coverage"]

println("\\nRunning uncertainty analysis...")
results = run_uncertainty_analysis(
    ground_truth,
    posterior_samples,
    output_dir,
    plots_to_generate,
    config
)

# Print results
println("\\nAnalysis Results:")
println("-"^30)
for (key, value) in results
    if isa(value, Number)
        println("$key: $(round(value, digits=4))")
    elseif isa(value, Vector) && length(value) <= 5
        println("$key: $value")
    end
end

println("\\n✓ Analysis complete! Check the '$output_dir' directory for results.")

# Example 3: Batch processing multiple datasets
println("\\n" * "="^60)
println("EXAMPLE 3: Batch Processing Template")
println("="^60)

println("""
# For processing multiple datasets:

datasets = [
    ("experiment_1", "data/gt_1.jld2", "data/post_1.jld2"),
    ("experiment_2", "data/gt_2.jld2", "data/post_2.jld2"),
    ("experiment_3", "data/gt_3.jld2", "data/post_3.jld2")
]

for (name, gt_path, post_path) in datasets
    println("Processing \$name...")
    
    # Load data
    gt = load_data(gt_path)
    post = load_data(post_path)
    
    # Run analysis
    output_dir = "./results/\$name/"
    results = run_uncertainty_analysis(gt, post, output_dir, ["all"], config)
    
    # Save results summary
    summary_path = joinpath(output_dir, "summary.jld2")
    JLD2.jldsave(summary_path; results...)
end
""")

println("\\n" * "="^60)
println("For more examples, see:")
println("- example_config.toml: Configuration file template")
println("- README.md: Complete documentation")
println("- src/: Source code with detailed function documentation")
println("="^60)