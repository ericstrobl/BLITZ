#' BLITZ conditional independence test
#'
#' Fast nonparametric conditional independence testing via two-stage residualization.
#'
#' @param x Numeric vector or matrix for X.
#' @param y Numeric vector or matrix for Y.
#' @param z Optional numeric vector or matrix of conditioning variables.
#' @param learning_rate Shrinkage applied to tree predictions.
#' @param maxdepth Maximum tree depth.
#' @param min_node_size Candidate minimum node sizes for tree tuning.
#' @param min_bucket Minimum child-node size.
#' @param mtry Number of conditioning variables considered at each split.
#' @param seed Optional random seed.
#' @param return_tuning Whether to return OOB tuning diagnostics.
#' @param poly_degree Polynomial degree for first-stage residualization.
#' @param poly_interactions Whether to include polynomial interactions.
#' @param poly_ridge Ridge regularization for polynomial regression.
#' @param subsample Bootstrap subsampling fraction for OOB tuning.
#'
#' @return A list containing the p-value, test statistic, and tuning information.
#' @export
BLITZ <- function(x, y, z = NULL,
                                     learning_rate = 1,
                                     maxdepth = 100,
                                     min_node_size = c(5L, 10L, 15L, 20L, 40L, 80L),
                                     min_bucket = 1L,
                                     mtry = NULL,
                                     seed = NULL,
                                     return_tuning = FALSE,
                                     poly_degree = 2L,
                                     poly_interactions = TRUE,
                                     poly_ridge = 1e-8,
                                     subsample = 1) {
  
  if (!requireNamespace("momentchi2", quietly = TRUE)) {
    stop("Need momentchi2. Install with install.packages('momentchi2').")
  }
  
  x_raw <- matrix2_double(x)
  y_raw <- matrix2_double(y)
  
  if (nrow(x_raw) != nrow(y_raw)) {
    stop("x and y must have the same number of rows.")
  }
  
  if (!all(is.finite(x_raw))) {
    stop("x contains non-finite values.")
  }
  
  if (!all(is.finite(y_raw))) {
    stop("y contains non-finite values.")
  }
  
  n <- nrow(x_raw)
  px <- ncol(x_raw)
  py <- ncol(y_raw)
  
  is_empty_z <- is.null(z) ||
    length(z) == 0L ||
    (!is.null(dim(z)) && length(dim(z)) == 2L && ncol(z) == 0L)
  
  z_pre <- NULL
  mtry_eff <- NULL
  
  if (!is_empty_z) {
    z <- matrix2_double(z)
    
    if (nrow(z) != n) {
      stop("z must have the same number of rows as x and y.")
    }
    
    if (!all(is.finite(z))) {
      stop("z contains non-finite values.")
    }
    
    z_sds <- fast_col_sds(z)
    z <- z[, z_sds > 0, drop = FALSE]
    
    if (ncol(z) == 0L) {
      is_empty_z <- TRUE
    } else {
      z_pre <- prep_Z_cpp_tree(z)
      mtry_eff <- if (is.null(mtry)) ncol(z_pre) else as.integer(mtry)
      
      raw_resid <- cpp_poly_residualize_matrix(
        Z = z_pre,
        A = cbind(x_raw, y_raw),
        degree = as.integer(poly_degree),
        interactions = isTRUE(poly_interactions),
        ridge = as.double(poly_ridge)
      )
      
      x_raw <- raw_resid[, seq_len(px), drop = FALSE]
      y_raw <- raw_resid[, px + seq_len(py), drop = FALSE]
    }
  }
  
  x <- asinh(x_raw)
  y <- asinh(y_raw)
  
  x_info <- normalize_fast_info(x)
  y_info <- normalize_fast_info(y)
  
  if (all(x_info$sd == 0) || all(y_info$sd == 0)) {
    return(list(
      p = 1,
      Sta = 0,
      nrounds_x = if (is_empty_z) 0L else 1L,
      nrounds_y = if (is_empty_z) 0L else 1L,
      min_node_size_x = NA_integer_,
      min_node_size_y = NA_integer_
    ))
  }
  
  x <- x_info$z
  y <- y_info$z
  
  gX <- swish_pair(x, "gX")
  fY <- swish_pair(y, "fY")
  
  min_node_tuning_x <- NULL
  min_node_tuning_y <- NULL
  
  if (is_empty_z) {
    resX <- sweep(gX, 2L, colMeans(gX), "-", check.margin = FALSE)
    resY <- sweep(fY, 2L, colMeans(fY), "-", check.margin = FALSE)
    
    min_node_size_x <- NA_integer_
    min_node_size_y <- NA_integer_
    
    nrounds_x <- 0L
    nrounds_y <- 0L
  } else {
    min_node_grid <- as.integer(min_node_size)
    min_node_grid <- unique(min_node_grid[!is.na(min_node_grid) & min_node_grid >= 1L])
    min_node_grid <- sort(min_node_grid)
    
    if (length(min_node_grid) == 0L) {
      stop("No valid min_node_size values.")
    }
    
    if (length(min_node_grid) > 1L) {
      min_node_tuning_x <- pick_min_node_size_oob_cpp(
        target_mat = gX,
        Z_pre = z_pre,
        min_node_candidates = min_node_grid,
        learning_rate = learning_rate,
        maxdepth = maxdepth,
        subsample = subsample,
        seed = if (is.null(seed)) NULL else seed + 50000L,
        mtry = mtry_eff,
        min_bucket = min_bucket
      )
      
      min_node_tuning_y <- pick_min_node_size_oob_cpp(
        target_mat = fY,
        Z_pre = z_pre,
        min_node_candidates = min_node_grid,
        learning_rate = learning_rate,
        maxdepth = maxdepth,
        subsample = subsample,
        seed = if (is.null(seed)) NULL else seed + 60000L,
        mtry = mtry_eff,
        min_bucket = min_bucket
      )
      
      min_node_size_x <- min_node_tuning_x$best_min_node_size
      min_node_size_y <- min_node_tuning_y$best_min_node_size
    } else {
      min_node_size_x <- min_node_grid[1L]
      min_node_size_y <- min_node_grid[1L]
    }
    
    nrounds_x <- 1L
    nrounds_y <- 1L
    
    resX <- cpp_cit_residualize_matrix_one(
      X = z_pre,
      Y = gX,
      min_node_by_col = as.integer(min_node_size_x),
      learning_rate = learning_rate,
      max_depth = as.integer(maxdepth),
      min_bucket = as.integer(min_bucket),
      mtry = if (is.null(mtry_eff)) 0L else as.integer(mtry_eff),
      seed_base = if (is.null(seed)) NA_integer_ else as.integer(seed)
    )
    
    resY <- cpp_cit_residualize_matrix_one(
      X = z_pre,
      Y = fY,
      min_node_by_col = as.integer(min_node_size_y),
      learning_rate = learning_rate,
      max_depth = as.integer(maxdepth),
      min_bucket = as.integer(min_bucket),
      mtry = if (is.null(mtry_eff)) 0L else as.integer(mtry_eff),
      seed_base = if (is.null(seed)) NA_integer_ else as.integer(seed + 1000L)
    )
  }
  
  Cxy_z <- stats::cov(resX, resY)
  Sta <- n * sum(Cxy_z^2)
  
  eig <- eig_product_cov_from_resids_fast(resX, resY)
  
  add_tuning_output <- function(out) {
    if (return_tuning && !is.null(min_node_tuning_x)) {
      out$oob_scores_x <- min_node_tuning_x$scores
      out$oob_score_se_x <- min_node_tuning_x$score_se
      out$oob_score_matrix_x <- min_node_tuning_x$score_mat
    }
    
    if (return_tuning && !is.null(min_node_tuning_y)) {
      out$oob_scores_y <- min_node_tuning_y$scores
      out$oob_score_se_y <- min_node_tuning_y$score_se
      out$oob_score_matrix_y <- min_node_tuning_y$score_mat
    }
    
    out
  }
  
  if (length(eig) == 0L) {
    out <- list(
      p = 1,
      Sta = Sta,
      nrounds_x = nrounds_x,
      nrounds_y = nrounds_y,
      min_node_size_x = min_node_size_x,
      min_node_size_y = min_node_size_y
    )
    
    return(add_tuning_output(out))
  }
  
  p <- try(1 - momentchi2::lpb4(eig, Sta), silent = TRUE)
  
  if (inherits(p, "try-error") ||
      !is.numeric(p) ||
      length(p) != 1L ||
      is.nan(p) ||
      is.na(p)) {
    p <- 1 - momentchi2::hbe(eig, Sta)
  }
  
  if (!is.numeric(p) || length(p) != 1L || is.nan(p) || is.na(p) || p < 0) {
    p <- 0
  }
  
  if (p > 1) {
    p <- 1
  }
  
  out <- list(
    p = p,
    Sta = Sta,
    nrounds_x = nrounds_x,
    nrounds_y = nrounds_y,
    min_node_size_x = min_node_size_x,
    min_node_size_y = min_node_size_y
  )
  
  add_tuning_output(out)
}
