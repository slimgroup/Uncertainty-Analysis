"""
Plotting functions for uncertainty analysis
"""

function setup_plot_style(config::UncertaintyAnalysisConfig)
    """Setup consistent plot styling"""
    PyPlot.rcdefaults()
    fontsize = 14
    PyPlot.rc("font", size=fontsize, family="serif")
    PyPlot.rc("axes", titlesize=fontsize, labelsize=fontsize)
    PyPlot.rc("xtick", labelsize=fontsize)
    PyPlot.rc("ytick", labelsize=fontsize)
    PyPlot.rc("legend", fontsize=fontsize)
    PyPlot.rc("figure", titlesize=fontsize)
end

function get_extent(data_size, grid_spacing)
    """Get extent for imshow plotting"""
    nx, nz = data_size
    dx, dz = grid_spacing
    return [0, (nx-1)*dx, (nz-1)*dz, 0]
end

function plot_x_gt(x_gt, output_dir, config; suffix="", plot_name="ground_truth")
    """Plot ground truth velocity model - adapted from utils_inference.jl"""
    setup_plot_style(config)
    
    extent = get_extent(size(x_gt), config.grid_spacing)
    
    fig, ax = subplots(figsize=(10, 6))
    im = ax.imshow(x_gt', vmin=minimum(x_gt), vmax=config.vmax_velocity, 
                   cmap="cet_rainbow4", extent=extent, aspect="auto", interpolation="none")
    ax.set_title("Ground Truth")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Depth [m]")
    
    cbar = colorbar(im, fraction=0.0235, pad=0.04)
    cbar.set_label("[km/s]")
    
    output_path = joinpath(output_dir, plot_name * suffix * ".$(config.output_format)")
    savefig(output_path, bbox_inches="tight", dpi=config.dpi)
    close(fig)
    
    config.verbose && println("  Saved: $output_path")
end

function plot_x_hat(X_post, x_gt, output_dir, config; X_post_type="posterior_samples", suffix="", plot_name="posterior_mean")
    """Plot posterior mean with SSIM - adapted from utils_inference.jl"""
    setup_plot_style(config)
    
    if X_post_type == "posterior_samples"
        X_post_mean = mean(X_post, dims=4)
        x_hat = Array(X_post_mean[:,:,1,1])
    elseif X_post_type == "posterior_mean"
        x_hat = X_post
    else
        X_post_mean = mean(X_post, dims=4)
        x_hat = Array(X_post_mean[:,:,1,1])
    end
    
    extent = get_extent(size(x_hat), config.grid_spacing)
    ssim = assess_ssim(x_hat, x_gt)
    
    fig, ax = subplots(figsize=(10, 6))
    im = ax.imshow(x_hat', vmin=minimum(x_gt), vmax=config.vmax_velocity, 
                   cmap="cet_rainbow4", extent=extent, aspect="auto", interpolation="none")
    ax.set_title("Posterior Mean | SSIM = " * string(round(ssim, digits=3)))
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Depth [m]")
    
    cbar = colorbar(im, fraction=0.0235, pad=0.04)
    cbar.set_label("[km/s]")
    
    output_path = joinpath(output_dir, plot_name * suffix * ".$(config.output_format)")
    savefig(output_path, bbox_inches="tight", dpi=config.dpi)
    close(fig)
    
    config.verbose && println("  Saved: $output_path")
    return ssim
end

function plot_posterior_mean(posterior_mean, ground_truth, output_dir, config, ssim_value)
    """Plot posterior mean velocity model"""
    plot_x_hat(reshape(posterior_mean, size(posterior_mean)..., 1, 1), ground_truth, 
               output_dir, config; X_post_type="posterior_mean", plot_name="posterior_mean")
end

function plot_error(X_post, x_gt, output_dir, config; X_post_type="posterior_samples", suffix="", plot_name="error")
    """Plot error map with RMSE - adapted from utils_inference.jl"""
    setup_plot_style(config)
    
    if X_post_type == "posterior_samples"
        X_post_mean = mean(X_post, dims=4)
        x_hat = Array(X_post_mean[:,:,1,1])
    elseif X_post_type == "posterior_mean"
        x_hat = X_post
    else
        X_post_mean = mean(X_post, dims=4)
        x_hat = Array(X_post_mean[:,:,1,1])
    end
    
    error_mean = abs.(x_hat - x_gt)
    rmse = round(sqrt(mean(error_mean.^2)), digits=3)
    extent = get_extent(size(X_post), config.grid_spacing)
    
    fig, ax = subplots(figsize=(10, 6))
    im = ax.imshow(error_mean', vmin=0, vmax=config.vmax_error, cmap="magma",
                   extent=extent, aspect="auto", interpolation="none")
    ax.set_title("Error | RMSE=$(rmse) [km/s]")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Depth [m]")
    
    cbar = colorbar(im, fraction=0.0235, pad=0.04)
    cbar.set_label("[km/s]")
    
    output_path = joinpath(output_dir, plot_name * suffix * ".$(config.output_format)")
    savefig(output_path, bbox_inches="tight", dpi=config.dpi)
    close(fig)
    
    config.verbose && println("  Saved: $output_path")
    return rmse
end

function plot_error_map(posterior_mean, ground_truth, output_dir, config, rmse_value)
    """Plot absolute error between posterior mean and ground truth"""
    plot_error(reshape(posterior_mean, size(posterior_mean)..., 1, 1), ground_truth, 
               output_dir, config; X_post_type="posterior_mean", plot_name="error_map")
end

function plot_rmse_map(posterior_samples, ground_truth, output_dir, config)
    """Plot pixel-wise RMSE map"""
    setup_plot_style(config)
    
    # Compute pixel-wise RMSE
    n_samples = size(posterior_samples, 4)
    squared_errors = zeros(size(ground_truth))
    
    for i in 1:n_samples
        sample = posterior_samples[:, :, 1, i]
        squared_errors .+= (sample - ground_truth).^2
    end
    rmse_map = sqrt.(squared_errors / n_samples)
    
    extent = get_extent(size(rmse_map), config.grid_spacing)
    
    fig, ax = subplots(figsize=(10, 6))
    im = ax.imshow(rmse_map', vmin=0, vmax=config.vmax_error, cmap="magma",
                   extent=extent, aspect="auto", interpolation="none")
    ax.set_title("Pixel-wise RMSE")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Depth [m]")
    
    cbar = colorbar(im, fraction=0.024, pad=0.02)
    cbar.set_label("[km/s]")
    
    output_path = joinpath(output_dir, "rmse_map.$(config.output_format)")
    savefig(output_path, bbox_inches="tight", dpi=config.dpi)
    close(fig)
    
    config.verbose && println("  Saved: $output_path")
end

function plot_uncert(X_post, output_dir, config; suffix="", plot_name="uncertainty")
    """Plot uncertainty (standard deviation) map - adapted from utils_inference.jl"""
    setup_plot_style(config)
    
    X_post_std = std(X_post, dims=4)
    stdtotal = round(sqrt(mean(X_post_std.^2)), digits=3)
    extent = get_extent(size(X_post), config.grid_spacing)
    
    fig, ax = subplots(figsize=(10, 6))
    im = ax.imshow(X_post_std[:,:,1,1]', vmin=0, vmax=config.vmax_uncertainty, cmap="magma",
                   extent=extent, aspect="auto", interpolation="none")
    ax.set_title("Variance | RMS pointwise STD="*string(stdtotal) * " [km/s]")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Z [m]")
    
    cbar = colorbar(im, fraction=0.024, pad=0.015)
    cbar.set_label("[km/s]")
    cbar.ax.tick_params(axis="both", length=1, width=1)
    
    output_path = joinpath(output_dir, plot_name * suffix * ".$(config.output_format)")
    savefig(output_path, bbox_inches="tight", dpi=config.dpi)
    close(fig)
    
    config.verbose && println("  Saved: $output_path")
    return stdtotal
end

function plot_uncertainty_map(posterior_std, output_dir, config, mean_uncertainty)
    """Plot uncertainty (standard deviation) map"""
    # Convert 2D std to 4D format for consistency
    X_post_std_4d = reshape(posterior_std, size(posterior_std)..., 1, 1)
    plot_uncert(X_post_std_4d, output_dir, config; plot_name="uncertainty_map")
end

function plot_calibration(uce, uncert_bin, errors_bin, output_dir, config; plot_name="calibration")
    """Plot uncertainty calibration curve - adapted from utils_inference.jl"""
    setup_plot_style(config)
    
    fig, ax = subplots(figsize=(7, 7))
    
    # Filter out NaN values
    valid_indices = .!isnan.(uncert_bin) .& .!isnan.(errors_bin)
    uncert_valid = uncert_bin[valid_indices]
    errors_valid = errors_bin[valid_indices]
    
    # Plot calibration curve
    ax.plot(uncert_valid, errors_valid, "-o", color="black", fillstyle="none",
            linewidth=0.9, label="UCE=$(round(uce, digits=2)) (km/s)")
    
    # Perfect calibration line
    if length(uncert_valid) > 0
        max_uncert_no_NaN = maximum(uncert_valid)
        max_errors_no_NaN = maximum(errors_valid)
        perfect_calibration_line = range(0, max_uncert_no_NaN+uncert_valid[1], length=100)
        ax.plot(perfect_calibration_line, perfect_calibration_line, "--", color="black", 
                label="Optimal calibration")
    end
    
    ax.set_aspect(1)
    ax.legend()
    ax.set_xlabel("Uncertainty (km/s)")
    ax.set_ylabel("Predictive error (km/s)")
    
    output_path = joinpath(output_dir, plot_name * ".$(config.output_format)")
    savefig(output_path, bbox_inches="tight", dpi=config.dpi)
    close(fig)
    
    config.verbose && println("  Saved: $output_path")
end

function plot_calibration_curve(uce, uncert_binned, errors_binned, output_dir, config)
    """Plot uncertainty calibration curve"""
    plot_calibration(uce, uncert_binned, errors_binned, output_dir, config; plot_name="calibration_curve")
end

function plot_z_score(z_score, perc_all, output_dir, config; plot_name="z_score")
    """
    Plot z-score map with threshold highlighting - adapted from utils_inference.jl lines 749-767
    """
    setup_plot_style(config)
    
    min_press, max_press = 0, 5
    crack_press = config.zscore_threshold
    
    # Create custom colormap (simplified version)
    extent = get_extent(size(z_score), config.grid_spacing)
    
    fig, ax = subplots(figsize=(10, 6))
    im = ax.imshow(z_score', vmin=0, vmax=max_press, cmap="viridis",
                   extent=extent, aspect="auto", interpolation="none")
    ax.set_title("Percent of error out of support of UQ $(round(perc_all, digits=1))%")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Depth [m]")
    
    cbar = colorbar(im, fraction=0.024, pad=0.02)
    cbar.set_label("Z-score")
    
    output_path = joinpath(output_dir, plot_name * ".$(config.output_format)")
    savefig(output_path, bbox_inches="tight", dpi=config.dpi)
    close(fig)
    
    config.verbose && println("  Saved: $output_path")
end

function plot_zscore_map(zscore_map, outlier_percentage, output_dir, config)
    """Plot z-score map with highlighted outliers"""
    plot_z_score(zscore_map, outlier_percentage, output_dir, config; plot_name="zscore_map")
end

function plot_coverage_analysis(coverage_stats, confidence_levels, output_dir, config)
    """Plot coverage analysis results"""
    setup_plot_style(config)
    
    fig, ax = subplots(figsize=(10, 6))
    
    # Convert to percentages
    confidence_pct = confidence_levels * 100
    coverage_pct = coverage_stats * 100
    
    # Bar plot
    bars = ax.bar(confidence_pct, coverage_pct, alpha=0.7, color="steelblue", 
                  edgecolor="black", linewidth=1)
    
    # Perfect coverage line
    ax.plot(confidence_pct, confidence_pct, "r--", linewidth=2, label="Perfect coverage")
    
    # Add value labels on bars
    for i, (conf, cov) in enumerate(zip(confidence_pct, coverage_pct))
        ax.text(conf, cov + 1, f"{cov:.1f}%", ha="center", va="bottom", fontweight="bold")
    
    ax.set_xlabel("Theoretical Confidence Level [%]")
    ax.set_ylabel("Empirical Coverage [%]")
    ax.set_title("Coverage Analysis")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set y-axis limits
    ax.set_ylim(0, 105)
    
    output_path = joinpath(output_dir, "coverage_analysis.$(config.output_format)")
    savefig(output_path, bbox_inches="tight", dpi=config.dpi)
    close(fig)
    
    config.verbose && println("  Saved: $output_path")
end

function plot_posterior(X_post, x_gt, output_dir, config; suffix="", test_idx=851, batch_size=64)
    """
    Comprehensive posterior plotting function - adapted from utils_inference.jl
    Generates ground truth, posterior mean, error, and uncertainty plots
    """
    X_post_mean = mean(X_post, dims=4)
    X_post_std = std(X_post, dims=4)
    
    x_hat = Array(X_post_mean[:,:,1,1])
    error_mean = abs.(x_hat - x_gt)
    
    ssim = round(assess_ssim(x_hat, x_gt), digits=3)
    rmse = round(sqrt(mean(error_mean.^2)), digits=3)
    stdtotal = round(sqrt(mean(X_post_std.^2)), digits=3)
    
    setup_plot_style(config)
    
    fname_x_gt = "idx=$(test_idx)_x_gt"
    fname_x_hat = "idx=$(test_idx)_x_hat" * suffix
    fname_error = "idx=$(test_idx)_error" * suffix
    fname_uncert = "idx=$(test_idx)_X_post_std" * suffix
    
    # Plot individual components
    plot_x_gt(x_gt, output_dir, config; plot_name=fname_x_gt)
    plot_x_hat(X_post, x_gt, output_dir, config; suffix="", plot_name=fname_x_hat)
    plot_error(X_post, x_gt, output_dir, config; suffix="", plot_name=fname_error)
    plot_uncert(X_post, output_dir, config; suffix="", plot_name=fname_uncert)
    
    close()
    
    return Dict("ssim" => ssim, "rmse" => rmse, "stdtotal" => stdtotal)
end