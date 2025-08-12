"""
Metric computation functions for uncertainty analysis
"""

function compute_basic_metrics(posterior_samples, ground_truth)
    """Compute basic uncertainty metrics"""
    posterior_mean = mean(posterior_samples, dims=4)[:, :, 1, 1]
    posterior_std = std(posterior_samples, dims=4)[:, :, 1, 1]
    
    # Structural similarity
    ssim = assess_ssim(posterior_mean, ground_truth)
    
    # Root mean square error
    rmse = sqrt(mean((posterior_mean - ground_truth).^2))
    
    # Mean absolute error
    mae = mean(abs.(posterior_mean - ground_truth))
    
    # Uncertainty metrics
    mean_uncertainty = mean(posterior_std)
    max_uncertainty = maximum(posterior_std)
    
    return Dict(
        "ssim" => ssim,
        "rmse" => rmse,
        "mae" => mae,
        "mean_uncertainty" => mean_uncertainty,
        "max_uncertainty" => max_uncertainty,
        "posterior_mean" => posterior_mean,
        "posterior_std" => posterior_std
    )
end

function compute_pixelwise_rmse(posterior_samples, ground_truth)
    """Compute pixel-wise RMSE across all posterior samples"""
    n_samples = size(posterior_samples, 4)
    squared_errors = zeros(size(ground_truth))
    
    for i in 1:n_samples
        sample = posterior_samples[:, :, 1, i]
        squared_errors .+= (sample - ground_truth).^2
    end
    
    return sqrt.(squared_errors / n_samples)
end

function compute_z_score(X_post, x_gt, threshold=2.0)
    """
    Compute z-score analysis - adapted from utils_inference.jl lines 727-746
    """
    # Compute posterior mean
    x_hat = mean(X_post, dims=4)[:, :, 1, 1]
    
    # Compute error against gt
    error_mean = x_hat - x_gt
    
    # Compute std
    X_post_std = std(X_post, dims=4)[:, :, 1, 1]
    
    # Compute z-score
    z_score = abs.(error_mean) ./ X_post_std
    
    # Compute percentage above threshold
    percentage_above_threshold = sum(z_score .> threshold) / length(z_score) * 100
    
    return z_score, percentage_above_threshold
end

function compute_zscore_analysis(posterior_samples, ground_truth, threshold=2.0)
    """Compute z-score analysis for outlier detection"""
    return compute_z_score(posterior_samples, ground_truth, threshold)
end

function compute_prediction_intervals(posterior_samples, confidence_levels)
    """Compute prediction intervals for given confidence levels"""
    intervals = Dict()
    
    for level in confidence_levels
        alpha = 1 - level
        lower_q = alpha / 2
        upper_q = 1 - alpha / 2
        
        lower_bound = mapslices(x -> quantile(x, lower_q), posterior_samples, dims=4)[:, :, 1, 1]
        upper_bound = mapslices(x -> quantile(x, upper_q), posterior_samples, dims=4)[:, :, 1, 1]
        
        intervals[level] = (lower_bound, upper_bound)
    end
    
    return intervals
end

function compute_sharpness_metrics(posterior_samples)
    """Compute sharpness metrics for uncertainty quantification"""
    posterior_std = std(posterior_samples, dims=4)[:, :, 1, 1]
    
    # Mean interval width (using 95% intervals as default)
    lower_bound = mapslices(x -> quantile(x, 0.025), posterior_samples, dims=4)[:, :, 1, 1]
    upper_bound = mapslices(x -> quantile(x, 0.975), posterior_samples, dims=4)[:, :, 1, 1]
    interval_width = upper_bound - lower_bound
    
    return Dict(
        "mean_std" => mean(posterior_std),
        "mean_interval_width" => mean(interval_width),
        "std_of_std" => std(posterior_std[:])
    )
end

function compute_reliability_metrics(posterior_samples, ground_truth, confidence_levels)
    """Compute reliability metrics across different confidence levels"""
    intervals = compute_prediction_intervals(posterior_samples, confidence_levels)
    reliability_metrics = Dict()
    
    for level in confidence_levels
        lower_bound, upper_bound = intervals[level]
        
        # Coverage: fraction of ground truth values within prediction intervals
        in_interval = (ground_truth .>= lower_bound) .& (ground_truth .<= upper_bound)
        empirical_coverage = mean(in_interval)
        
        # Interval score (lower is better)
        alpha = 1 - level
        interval_score = mean(
            (upper_bound - lower_bound) +
            (2/alpha) * (lower_bound - ground_truth) .* (ground_truth .< lower_bound) +
            (2/alpha) * (ground_truth - upper_bound) .* (ground_truth .> upper_bound)
        )
        
        reliability_metrics[level] = Dict(
            "coverage" => empirical_coverage,
            "interval_score" => interval_score,
            "mean_interval_width" => mean(upper_bound - lower_bound)
        )
    end
    
    return reliability_metrics
end