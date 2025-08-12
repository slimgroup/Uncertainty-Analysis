"""
Coverage analysis functions for uncertainty quantification
"""

function compute_coverage_statistics(posterior_samples, ground_truth, confidence_levels)
    """
    Compute empirical coverage statistics for given confidence levels
    
    Based on the implementation in utils_inference.jl compute_coverage function
    """
    coverage_stats = Float64[]
    
    for level in confidence_levels
        alpha = 1 - level
        lower_percentile = alpha / 2
        upper_percentile = 1 - alpha / 2
        
        coverage = compute_empirical_coverage(posterior_samples, ground_truth, 
                                            lower_percentile, upper_percentile)
        push!(coverage_stats, coverage)
    end
    
    return coverage_stats
end

function compute_coverage(X_post, x_gt; confidence_level=0.98)
    """
    Compute coverage statistics - adapted from utils_inference.jl lines 565-590
    """
    # Initialize the count for pixels falling within the percentile range
    in_range_count = 0
    
    # Calculate percentiles based on confidence level
    alpha = 1 - confidence_level
    p_lower = alpha / 2
    p_upper = 1 - alpha / 2
    
    nx, nz = size(x_gt)
    
    # Iterate over each pixel
    for i in 1:nx, j in 1:nz
        # Extract the posterior samples for this pixel (i, j)
        samples = X_post[i, j, 1, :]
        
        # Calculate the lower and upper percentiles
        p1 = quantile(samples, p_lower)
        p99 = quantile(samples, p_upper)
        
        # Check if the ground truth falls within the percentile range
        if x_gt[i, j] >= p1 && x_gt[i, j] <= p99
            in_range_count += 1
        end
    end
    
    # Calculate the percentage of pixels that fall within the range
    total_pixels = nx * nz
    percentage_in_range = (in_range_count / total_pixels) * 100
    
    println("Percentage of pixels within the $(Int(confidence_level*100))% percentile range: $percentage_in_range%")
    return percentage_in_range
end

function compute_empirical_coverage(posterior_samples, ground_truth, lower_percentile, upper_percentile)
    """
    Compute empirical coverage for a specific confidence interval
    """
    confidence_level = upper_percentile - lower_percentile
    return compute_coverage(posterior_samples, ground_truth; confidence_level=confidence_level) / 100
end

function compute_coverage_map(posterior_samples, ground_truth, confidence_level=0.95)
    """
    Compute a spatial map showing which pixels have ground truth within confidence intervals
    """
    alpha = 1 - confidence_level
    lower_percentile = alpha / 2
    upper_percentile = 1 - alpha / 2
    
    nx, nz = size(ground_truth)
    coverage_map = zeros(Bool, nx, nz)
    
    for i in 1:nx, j in 1:nz
        samples = posterior_samples[i, j, 1, :]
        lower_bound = quantile(samples, lower_percentile)
        upper_bound = quantile(samples, upper_percentile)
        
        coverage_map[i, j] = (ground_truth[i, j] >= lower_bound) && (ground_truth[i, j] <= upper_bound)
    end
    
    return coverage_map
end

function compute_interval_widths(posterior_samples, confidence_levels)
    """
    Compute prediction interval widths for different confidence levels
    """
    interval_widths = Dict()
    
    for level in confidence_levels
        alpha = 1 - level
        lower_percentile = alpha / 2
        upper_percentile = 1 - alpha / 2
        
        lower_bounds = mapslices(x -> quantile(x, lower_percentile), posterior_samples, dims=4)[:, :, 1, 1]
        upper_bounds = mapslices(x -> quantile(x, upper_percentile), posterior_samples, dims=4)[:, :, 1, 1]
        
        widths = upper_bounds - lower_bounds
        interval_widths[level] = widths
    end
    
    return interval_widths
end

function compute_coverage_width_relationship(posterior_samples, ground_truth, confidence_levels)
    """
    Analyze the relationship between coverage and interval width
    """
    results = Dict()
    
    for level in confidence_levels
        # Compute coverage
        coverage = compute_empirical_coverage(
            posterior_samples, ground_truth, 
            (1-level)/2, 1-(1-level)/2
        )
        
        # Compute mean interval width
        interval_widths = compute_interval_widths(posterior_samples, [level])
        mean_width = mean(interval_widths[level])
        
        results[level] = Dict(
            "coverage" => coverage,
            "mean_width" => mean_width,
            "theoretical_coverage" => level
        )
    end
    
    return results
end

function compute_conditional_coverage(posterior_samples, ground_truth, condition_map, confidence_level=0.95)
    """
    Compute coverage statistics conditioned on a spatial mask
    
    Args:
        condition_map: Boolean array of same size as ground_truth indicating regions of interest
    """
    alpha = 1 - confidence_level
    lower_percentile = alpha / 2
    upper_percentile = 1 - alpha / 2
    
    # Get indices where condition is true
    condition_indices = findall(condition_map)
    
    if length(condition_indices) == 0
        return 0.0
    end
    
    in_range_count = 0
    
    for idx in condition_indices
        i, j = Tuple(idx)
        samples = posterior_samples[i, j, 1, :]
        
        lower_bound = quantile(samples, lower_percentile)
        upper_bound = quantile(samples, upper_percentile)
        
        if ground_truth[i, j] >= lower_bound && ground_truth[i, j] <= upper_bound
            in_range_count += 1
        end
    end
    
    return in_range_count / length(condition_indices)
end

function assess_coverage_calibration(posterior_samples, ground_truth, confidence_levels)
    """
    Assess how well the empirical coverage matches the theoretical coverage
    """
    empirical_coverage = compute_coverage_statistics(posterior_samples, ground_truth, confidence_levels)
    
    coverage_gaps = abs.(empirical_coverage - confidence_levels)
    mean_coverage_gap = mean(coverage_gaps)
    max_coverage_gap = maximum(coverage_gaps)
    
    return Dict(
        "empirical_coverage" => empirical_coverage,
        "theoretical_coverage" => confidence_levels,
        "coverage_gaps" => coverage_gaps,
        "mean_coverage_gap" => mean_coverage_gap,
        "max_coverage_gap" => max_coverage_gap
    )
end