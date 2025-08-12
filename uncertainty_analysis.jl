#!/usr/bin/env julia

"""
Uncertainty Analysis CLI Tool

A command-line tool for analyzing uncertainty in posterior samples from Bayesian inverse problems.

Usage:
    julia uncertainty_analysis.jl --ground-truth path/to/ground_truth.jld2 --posterior-samples path/to/posterior_samples.jld2 --output results/
    julia uncertainty_analysis.jl --config config.toml

Examples:
    # Basic usage with individual files
    julia uncertainty_analysis.jl --ground-truth velocity_gt.jld2 --posterior-samples velocity_post.jld2 --output ./results/

    # Using configuration file
    julia uncertainty_analysis.jl --config uncertainty_config.toml

    # Specify which plots to generate
    julia uncertainty_analysis.jl --ground-truth gt.jld2 --posterior-samples post.jld2 --plots posterior_mean,error,calibration --output ./plots/
"""

using ArgParse
using JLD2
using NPZ
using TOML
include("src/UncertaintyAnalysis.jl")
using .UncertaintyAnalysis

function parse_commandline()
    s = ArgParseSettings(
        prog = "uncertainty_analysis.jl",
        description = "Uncertainty Analysis CLI Tool for Bayesian inverse problems",
        version = "0.1.0",
        add_version = true
    )

    @add_arg_table! s begin
        "--ground-truth", "-g"
            help = "Path to ground truth data file (.jld2, .npy)"
            arg_type = String
            
        "--posterior-samples", "-p"
            help = "Path to posterior samples file (.jld2, .npy)"
            arg_type = String
            
        "--posterior-mean", "-m"
            help = "Path to posterior mean file (if different from samples)"
            arg_type = String
            
        "--output", "-o"
            help = "Output directory for plots and results"
            arg_type = String
            default = "./uncertainty_results/"
            
        "--config", "-c"
            help = "Configuration file (.toml)"
            arg_type = String
            
        "--plots"
            help = "Comma-separated list of plots to generate: posterior_mean,error,rmse,calibration,zscore,coverage,all"
            arg_type = String
            default = "all"
            
        "--grid-spacing"
            help = "Grid spacing as 'dx,dz' in meters (e.g., '12.5,12.5')"
            arg_type = String
            default = "12.5,12.5"
            
        "--confidence-levels"
            help = "Confidence levels for coverage analysis (e.g., '0.68,0.95')"
            arg_type = String
            default = "0.68,0.95"
            
        "--calibration-bins"
            help = "Number of bins for calibration analysis"
            arg_type = Int
            default = 45
            
        "--zscore-threshold"
            help = "Z-score threshold for outlier detection"
            arg_type = Float64
            default = 2.0
            
        "--vmax-velocity"
            help = "Maximum velocity for color scale (km/s)"
            arg_type = Float64
            default = 4.5
            
        "--vmax-error"
            help = "Maximum error for color scale (km/s)"
            arg_type = Float64
            default = 0.42
            
        "--vmax-uncertainty"
            help = "Maximum uncertainty for color scale (km/s)"
            arg_type = Float64
            default = 0.42
            
        "--format"
            help = "Output format for plots (png, pdf, svg)"
            arg_type = String
            default = "png"
            
        "--dpi"
            help = "DPI for raster output formats"
            arg_type = Int
            default = 300
            
        "--verbose", "-v"
            help = "Verbose output"
            action = :store_true
    end

    return parse_args(s)
end

function load_config(config_path::String)
    """Load configuration from TOML file"""
    if !isfile(config_path)
        error("Configuration file not found: $config_path")
    end
    
    config = TOML.parsefile(config_path)
    return config
end

function parse_plots(plots_str::String)
    """Parse comma-separated plot list"""
    if plots_str == "all"
        return ["posterior_mean", "error", "rmse", "calibration", "zscore", "coverage"]
    else
        return split(plots_str, ",")
    end
end

function parse_grid_spacing(spacing_str::String)
    """Parse grid spacing string to tuple"""
    parts = split(spacing_str, ",")
    if length(parts) != 2
        error("Grid spacing must be specified as 'dx,dz'")
    end
    return (parse(Float64, parts[1]), parse(Float64, parts[2]))
end

function parse_confidence_levels(levels_str::String)
    """Parse confidence levels string to array"""
    parts = split(levels_str, ",")
    return [parse(Float64, level) for level in parts]
end

function load_data(filepath::String, variable_name::String="data")
    """Load data from various file formats"""
    if !isfile(filepath)
        error("File not found: $filepath")
    end
    
    ext = splitext(filepath)[2]
    
    if ext == ".jld2"
        data_dict = JLD2.jldopen(filepath, "r")
        # Try common variable names
        possible_names = [variable_name, "data", "m", "x", "velocity", "model"]
        for name in possible_names
            if haskey(data_dict, name)
                return data_dict[name]
            end
        end
        # If no common names found, list available keys
        keys_available = collect(keys(data_dict))
        error("Could not find data in $filepath. Available keys: $keys_available")
        
    elseif ext == ".npy"
        return npzread(filepath)
        
    else
        error("Unsupported file format: $ext. Supported formats: .jld2, .npy")
    end
end

function validate_data_dimensions(ground_truth, posterior_samples)
    """Validate that data dimensions are compatible"""
    gt_size = size(ground_truth)
    post_size = size(posterior_samples)
    
    # Ground truth should be 2D or 3D, posterior samples should be 4D
    if length(gt_size) < 2 || length(gt_size) > 3
        error("Ground truth should be 2D (nx, nz) or 3D (nx, nz, 1). Got size: $gt_size")
    end
    
    if length(post_size) != 4
        error("Posterior samples should be 4D (nx, nz, 1, n_samples). Got size: $post_size")
    end
    
    # Check spatial dimensions match
    if gt_size[1] != post_size[1] || gt_size[2] != post_size[2]
        error("Spatial dimensions mismatch. Ground truth: $(gt_size[1:2]), Posterior: $(post_size[1:2])")
    end
    
    println("✓ Data validation passed")
    println("  Ground truth size: $gt_size")
    println("  Posterior samples size: $post_size")
    println("  Number of posterior samples: $(post_size[4])")
end

function main()
    args = parse_commandline()
    
    # Load configuration if provided
    if args["config"] !== nothing
        config = load_config(args["config"])
        # Override command line args with config values
        for (key, value) in config
            if haskey(args, key) && value !== nothing
                args[key] = value
            end
        end
    end
    
    # Validate required arguments
    if args["ground-truth"] === nothing || args["posterior-samples"] === nothing
        error("Both --ground-truth and --posterior-samples are required")
    end
    
    # Create output directory
    output_dir = args["output"]
    if !isdir(output_dir)
        mkpath(output_dir)
        println("Created output directory: $output_dir")
    end
    
    # Parse arguments
    plots_to_generate = parse_plots(args["plots"])
    grid_spacing = parse_grid_spacing(args["grid-spacing"])
    confidence_levels = parse_confidence_levels(args["confidence-levels"])
    
    if args["verbose"]
        println("Arguments parsed:")
        println("  Ground truth: $(args["ground-truth"])")
        println("  Posterior samples: $(args["posterior-samples"])")
        println("  Output directory: $output_dir")
        println("  Plots to generate: $plots_to_generate")
        println("  Grid spacing: $grid_spacing")
    end
    
    # Load data
    println("Loading data...")
    ground_truth = load_data(args["ground-truth"])
    posterior_samples = load_data(args["posterior-samples"])
    
    # Load posterior mean if provided separately
    posterior_mean = nothing
    if args["posterior-mean"] !== nothing
        posterior_mean = load_data(args["posterior-mean"])
    end
    
    # Validate data
    validate_data_dimensions(ground_truth, posterior_samples)
    
    # Create analysis configuration
    analysis_config = UncertaintyAnalysisConfig(
        grid_spacing = grid_spacing,
        confidence_levels = confidence_levels,
        calibration_bins = args["calibration-bins"],
        zscore_threshold = args["zscore-threshold"],
        vmax_velocity = args["vmax-velocity"],
        vmax_error = args["vmax-error"],
        vmax_uncertainty = args["vmax-uncertainty"],
        output_format = args["format"],
        dpi = args["dpi"],
        verbose = args["verbose"]
    )
    
    # Run uncertainty analysis
    println("Starting uncertainty analysis...")
    results = run_uncertainty_analysis(
        ground_truth,
        posterior_samples,
        output_dir,
        plots_to_generate,
        analysis_config;
        posterior_mean = posterior_mean
    )
    
    # Save summary results
    summary_path = joinpath(output_dir, "analysis_summary.jld2")
    JLD2.jldsave(summary_path; results...)
    
    println("✓ Analysis complete! Results saved to: $output_dir")
    println("  Summary file: $summary_path")
    
    # Print key metrics
    if haskey(results, "ssim")
        println("  SSIM: $(round(results["ssim"], digits=3))")
    end
    if haskey(results, "rmse")
        println("  RMSE: $(round(results["rmse"], digits=3)) km/s")
    end
    if haskey(results, "uce")
        println("  UCE: $(round(results["uce"], digits=3))")
    end
    if haskey(results, "coverage_stats")
        for (level, coverage) in zip(confidence_levels, results["coverage_stats"])
            println("  Coverage $(Int(level*100))%: $(round(coverage*100, digits=1))%")
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end