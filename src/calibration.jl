"""
Uncertainty calibration analysis functions
"""

function collect_uncert_err_data(X_post, x_gt, n_bins)
    """
    Collect uncertainty and error data for calibration - adapted from utils_inference.jl lines 299-313
    """
    X_post_all = X_post
    X_post_mean_all = mean(X_post_all, dims=4)
    X_post_std_all = std(X_post_all, dims=4)
    x_hat_all = X_post_mean_all[:,:,1,1]
    error_mean_all = abs.(x_hat_all - x_gt)
    
    min_uncert = 0
    max_uncert = 1
    uce_all, errors_bin_all, uncert_bin_all, prop_in_bin_list_all = 
        uceloss(error_mean_all, X_post_std_all[:,:,1,1]; range=[min_uncert,max_uncert], n_bins=n_bins)
    
    return uce_all, errors_bin_all, uncert_bin_all
end

function compute_calibration(test_idx, output_dir; X_post=nothing, x_gt=nothing, batch_size=64, n_bins=64)
    """
    Compute and plot calibration of uncertainty - adapted from utils_inference.jl lines 390-409
    """
    println("test_idx: ", test_idx)
    
    uce_all, errors_bin_all, uncert_bin_all = collect_uncert_err_data(X_post, x_gt, n_bins)
    println("finish uce, errors, uncert")
    
    # Save results (optional)
    # JLD2.@save "$(output_dir)/errors_bin_all_idx=$(test_idx).jld2" errors_bin_all
    # JLD2.@save "$(output_dir)/uncert_bin_all_idx=$(test_idx).jld2" uncert_bin_all
    # JLD2.@save "$(output_dir)/uce_all_idx=$(test_idx).jld2" uce_all
    
    return uce_all, errors_bin_all, uncert_bin_all
end

function compute_calibration_metrics(posterior_samples, ground_truth, n_bins=45)
    """
    Compute uncertainty calibration error (UCE) and binned data for calibration plots
    
    Based on the implementation in utils_inference.jl uceloss function
    """
    # Use the adapted function
    return collect_uncert_err_data(posterior_samples, ground_truth, n_bins)
end

function uceloss(errors, uncert; n_bins=45, outlier=0.0, range=nothing)
    """
    Compute Uncertainty Calibration Error (UCE)
    
    Adapted from utils_inference.jl
    """
    if isnothing(range)
        bin_boundaries = LinRange(minimum(uncert), maximum(uncert), n_bins + 1)
    else
        bin_boundaries = LinRange(range[1], range[2], n_bins + 1)
    end
    
    bin_boundaries = collect(bin_boundaries)
    bin_lowers = bin_boundaries[1:end-1]
    bin_uppers = bin_boundaries[2:end]
    
    errors_in_bin_list = Float64[]
    avg_uncert_in_bin_list = Float64[]
    prop_in_bin_list = Float64[]
    
    uce = 0.0
    
    for (bin_lower, bin_upper) in zip(bin_lowers, bin_uppers)
        # Find points in this bin
        in_bin = (uncert .> bin_lower) .& (uncert .<= bin_upper)
        prop_in_bin = mean(in_bin)
        
        push!(prop_in_bin_list, prop_in_bin)
        
        if prop_in_bin > outlier
            # Compute average error and uncertainty in this bin
            errors_in_bin = mean(errors[in_bin])
            avg_uncert_in_bin = mean(uncert[in_bin])
            
            # Add to UCE
            uce += abs(avg_uncert_in_bin - errors_in_bin)
            
            push!(errors_in_bin_list, errors_in_bin)
            push!(avg_uncert_in_bin_list, avg_uncert_in_bin)
        else
            # Empty bin
            push!(errors_in_bin_list, NaN)
            push!(avg_uncert_in_bin_list, NaN)
        end
    end
    
    return uce / n_bins, errors_in_bin_list, avg_uncert_in_bin_list
end

function compute_calibration_curve_data(posterior_samples, ground_truth, n_bins=45)
    """
    Compute data for reliability diagram (calibration curve)
    """
    posterior_mean = mean(posterior_samples, dims=4)[:, :, 1, 1]
    posterior_std = std(posterior_samples, dims=4)[:, :, 1, 1]
    
    # Flatten for easier processing
    mean_flat = vec(posterior_mean)
    std_flat = vec(posterior_std)
    gt_flat = vec(ground_truth)
    
    # Create bins based on predicted uncertainty
    bin_edges = quantile(std_flat, range(0, 1, length=n_bins+1))
    bin_centers = (bin_edges[1:end-1] + bin_edges[2:end]) / 2
    
    observed_errors = Float64[]
    predicted_uncertainties = Float64[]
    bin_counts = Int[]
    
    for i in 1:n_bins
        if i == n_bins
            # Include the maximum value in the last bin
            in_bin = (std_flat .>= bin_edges[i]) .& (std_flat .<= bin_edges[i+1])
        else
            in_bin = (std_flat .>= bin_edges[i]) .& (std_flat .< bin_edges[i+1])
        end
        
        if sum(in_bin) > 0
            # Compute observed error in this bin
            bin_errors = abs.(mean_flat[in_bin] - gt_flat[in_bin])
            observed_error = mean(bin_errors)
            
            # Predicted uncertainty is the mean uncertainty in this bin
            predicted_uncertainty = mean(std_flat[in_bin])
            
            push!(observed_errors, observed_error)
            push!(predicted_uncertainties, predicted_uncertainty)
            push!(bin_counts, sum(in_bin))
        else
            push!(observed_errors, NaN)
            push!(predicted_uncertainties, NaN)
            push!(bin_counts, 0)
        end
    end
    
    return predicted_uncertainties, observed_errors, bin_counts
end

function compute_expected_calibration_error(posterior_samples, ground_truth, n_bins=45)
    """
    Compute Expected Calibration Error (ECE) for uncertainty quantification
    """
    predicted_uncertainties, observed_errors, bin_counts = compute_calibration_curve_data(
        posterior_samples, ground_truth, n_bins
    )
    
    total_samples = sum(bin_counts)
    ece = 0.0
    
    for i in 1:length(predicted_uncertainties)
        if !isnan(predicted_uncertainties[i]) && bin_counts[i] > 0
            weight = bin_counts[i] / total_samples
            calibration_error = abs(predicted_uncertainties[i] - observed_errors[i])
            ece += weight * calibration_error
        end
    end
    
    return ece
end

function compute_maximum_calibration_error(posterior_samples, ground_truth, n_bins=45)
    """
    Compute Maximum Calibration Error (MCE) for uncertainty quantification
    """
    predicted_uncertainties, observed_errors, bin_counts = compute_calibration_curve_data(
        posterior_samples, ground_truth, n_bins
    )
    
    max_error = 0.0
    
    for i in 1:length(predicted_uncertainties)
        if !isnan(predicted_uncertainties[i]) && bin_counts[i] > 0
            calibration_error = abs(predicted_uncertainties[i] - observed_errors[i])
            max_error = max(max_error, calibration_error)
        end
    end
    
    return max_error
end