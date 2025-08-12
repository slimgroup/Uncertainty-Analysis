"""
Utility functions for uncertainty analysis
"""

function validate_input_dimensions(ground_truth, posterior_samples)
    """
    Validate that input data has correct dimensions and compatibility
    """
    gt_size = size(ground_truth)
    post_size = size(posterior_samples)
    
    # Check ground truth dimensions
    if length(gt_size) < 2 || length(gt_size) > 3
        throw(ArgumentError("Ground truth should be 2D (nx, nz) or 3D (nx, nz, 1). Got size: $gt_size"))
    end
    
    # Check posterior samples dimensions
    if length(post_size) != 4
        throw(ArgumentError("Posterior samples should be 4D (nx, nz, 1, n_samples). Got size: $post_size"))
    end
    
    # Check spatial dimensions compatibility
    gt_spatial = gt_size[1:2]
    post_spatial = post_size[1:2]
    
    if gt_spatial != post_spatial
        throw(ArgumentError("Spatial dimensions mismatch. Ground truth: $gt_spatial, Posterior: $post_spatial"))
    end
    
    # Check minimum number of samples
    n_samples = post_size[4]
    if n_samples < 2
        throw(ArgumentError("Need at least 2 posterior samples for uncertainty analysis. Got: $n_samples"))
    end
    
    return true
end

function normalize_data_range(data; percentile=99.9)
    """
    Normalize data to [0, 1] range using percentile-based scaling
    """
    data_vec = vec(data)
    vmin = minimum(data_vec)
    vmax = quantile(data_vec, percentile/100)
    
    normalized = (data .- vmin) ./ (vmax - vmin)
    return clamp.(normalized, 0, 1), (vmin, vmax)
end

function load_data_flexible(filepath::String)
    """
    Load data from various formats with flexible key detection
    """
    if !isfile(filepath)
        throw(ArgumentError("File not found: $filepath"))
    end
    
    ext = lowercase(splitext(filepath)[2])
    
    if ext == ".jld2"
        data_dict = JLD2.jldopen(filepath, "r")
        
        # Common variable names to try
        possible_keys = [
            "data", "velocity", "model", "m", "x", "samples", 
            "posterior", "ground_truth", "gt", "mean", "std"
        ]
        
        # Try to find data automatically
        dict_keys = collect(keys(data_dict))
        
        for key in possible_keys
            if key in dict_keys
                return data_dict[key], key
            end
        end
        
        # If no common keys found, return first numeric array
        for key in dict_keys
            data = data_dict[key]
            if isa(data, AbstractArray) && eltype(data) <: Number
                @warn "Using key '$key' as no standard key found"
                return data, key
            end
        end
        
        throw(ArgumentError("No suitable numeric data found in $filepath. Available keys: $dict_keys"))
        
    elseif ext == ".npy"
        return npzread(filepath), "array"
        
    else
        throw(ArgumentError("Unsupported file format: $ext. Supported: .jld2, .npy"))
    end
end

function create_output_directory(base_path::String, create_subdirs=true)
    """
    Create output directory structure for uncertainty analysis results
    """
    if !isdir(base_path)
        mkpath(base_path)
    end
    
    if create_subdirs
        subdirs = ["plots", "data", "metrics"]
        for subdir in subdirs
            subdir_path = joinpath(base_path, subdir)
            if !isdir(subdir_path)
                mkpath(subdir_path)
            end
        end
    end
    
    return base_path
end

function save_analysis_config(config::UncertaintyAnalysisConfig, output_dir::String)
    """
    Save analysis configuration to file for reproducibility
    """
    config_dict = Dict(
        "grid_spacing" => config.grid_spacing,
        "confidence_levels" => config.confidence_levels,
        "calibration_bins" => config.calibration_bins,
        "zscore_threshold" => config.zscore_threshold,
        "vmax_velocity" => config.vmax_velocity,
        "vmax_error" => config.vmax_error,
        "vmax_uncertainty" => config.vmax_uncertainty,
        "output_format" => config.output_format,
        "dpi" => config.dpi,
        "timestamp" => string(now())
    )
    
    config_path = joinpath(output_dir, "analysis_config.jld2")
    JLD2.jldsave(config_path; config=config_dict)
    
    return config_path
end

function compute_data_statistics(data)
    """
    Compute basic statistics for data arrays
    """
    data_vec = vec(data)
    
    return Dict(
        "mean" => mean(data_vec),
        "std" => std(data_vec),
        "min" => minimum(data_vec),
        "max" => maximum(data_vec),
        "median" => median(data_vec),
        "q25" => quantile(data_vec, 0.25),
        "q75" => quantile(data_vec, 0.75),
        "q95" => quantile(data_vec, 0.95),
        "q99" => quantile(data_vec, 0.99)
    )
end

function format_scientific(value::Float64, digits=3)
    """
    Format number in scientific notation with specified digits
    """
    if abs(value) < 1e-3 || abs(value) >= 1e3
        return @sprintf("%.$(digits)e", value)
    else
        return @sprintf("%.$(digits)f", value)
    end
end

function create_summary_report(results::Dict, output_dir::String)
    """
    Create a text summary report of the uncertainty analysis
    """
    report_path = joinpath(output_dir, "uncertainty_analysis_report.txt")
    
    open(report_path, "w") do io
        println(io, "="^60)
        println(io, "UNCERTAINTY ANALYSIS REPORT")
        println(io, "="^60)
        println(io, "Generated: $(now())")
        println(io)
        
        # Basic metrics
        if haskey(results, "ssim")
            println(io, "BASIC METRICS:")
            println(io, "-"^20)
            println(io, "SSIM: $(format_scientific(results["ssim"]))")
        end
        
        if haskey(results, "rmse")
            println(io, "RMSE: $(format_scientific(results["rmse"])) km/s")
        end
        
        if haskey(results, "mean_uncertainty")
            println(io, "Mean Uncertainty: $(format_scientific(results["mean_uncertainty"])) km/s")
        end
        
        if haskey(results, "max_uncertainty")
            println(io, "Max Uncertainty: $(format_scientific(results["max_uncertainty"])) km/s")
        end
        
        println(io)
        
        # Calibration metrics
        if haskey(results, "uce")
            println(io, "CALIBRATION METRICS:")
            println(io, "-"^20)
            println(io, "UCE: $(format_scientific(results["uce"]))")
            println(io)
        end
        
        # Coverage statistics
        if haskey(results, "coverage_stats")
            println(io, "COVERAGE ANALYSIS:")
            println(io, "-"^20)
            coverage_stats = results["coverage_stats"]
            confidence_levels = get(results, "confidence_levels", [0.68, 0.95])
            
            for (level, coverage) in zip(confidence_levels, coverage_stats)
                println(io, "$(Int(level*100))% Confidence: $(round(coverage*100, digits=1))% empirical coverage")
            end
            println(io)
        end
        
        # Z-score analysis
        if haskey(results, "outlier_percentage")
            println(io, "Z-SCORE ANALYSIS:")
            println(io, "-"^20)
            println(io, "Outlier percentage: $(round(results["outlier_percentage"], digits=1))%")
            println(io)
        end
        
        println(io, "="^60)
    end
    
    return report_path
end