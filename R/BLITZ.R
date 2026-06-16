matrix2 <- function(a) {
  if (is.null(dim(a))) matrix(a, ncol = 1L) else as.matrix(a)
}

matrix2_double <- function(a) {
  a <- matrix2(a)
  if (!is.double(a)) storage.mode(a) <- "double"
  a
}

.check_cpp_cit_loaded_min <- function() {
  needed <- c(
    "cpp_poly_residualize_matrix",
    "cpp_cit_oob_score_min_node_matrix",
    "cpp_cit_residualize_matrix_one"
  )
  
  missing <- needed[!vapply(
    needed,
    exists,
    logical(1L),
    mode = "function",
    inherits = TRUE
  )]
  
  if (length(missing) > 0L) {
    stop(
      "Missing C++ functions: ",
      paste(missing, collapse = ", "),
      ". Run the corrected Rcpp::sourceCpp(...) block first."
    )
  }
}

fast_col_sds <- function(a) {
  a <- matrix2_double(a)
  
  n <- nrow(a)
  if (n <= 1L) return(rep(0, ncol(a)))
  
  mu <- colMeans(a)
  ac <- sweep(a, 2L, mu, "-", check.margin = FALSE)
  sqrt(colSums(ac * ac) / (n - 1L))
}

normalize_fast_info <- function(a) {
  a <- matrix2_double(a)
  
  n <- nrow(a)
  mu <- colMeans(a)
  ac <- sweep(a, 2L, mu, "-", check.margin = FALSE)
  
  if (n <= 1L) {
    s_raw <- rep(0, ncol(a))
    s <- rep(1, ncol(a))
  } else {
    s_raw <- sqrt(colSums(ac * ac) / (n - 1L))
    s <- s_raw
    s[!is.finite(s) | s == 0] <- 1
  }
  
  list(
    z = sweep(ac, 2L, s, "/", check.margin = FALSE),
    sd = s_raw
  )
}

prep_Z_cpp_tree <- function(Z) {
  Z <- matrix2_double(Z)
  
  if (is.null(colnames(Z))) {
    colnames(Z) <- paste0("V", seq_len(ncol(Z)))
  }
  
  Z
}

swish_pair <- function(a, prefix) {
  out <- cbind((-a) * plogis(-a), a * plogis(a))
  colnames(out) <- paste0(prefix, seq_len(ncol(out)))
  out
}

prod_features_fast <- function(A, B) {
  n <- nrow(A)
  p <- ncol(A)
  q <- ncol(B)
  
  P <- matrix(0, n, p * q)
  
  for (j in seq_len(q)) {
    cols <- ((j - 1L) * p + 1L):(j * p)
    P[, cols] <- A * B[, j]
  }
  
  P
}

eig_product_cov_from_resids_fast <- function(A, B) {
  n <- nrow(A)
  p <- ncol(A)
  q <- ncol(B)
  d <- p * q
  
  M <- if (d <= n) {
    P <- prod_features_fast(A, B)
    crossprod(P) / n
  } else {
    (tcrossprod(A) * tcrossprod(B)) / n
  }
  
  eig <- eigen(M, symmetric = TRUE, only.values = TRUE)$values
  eig[is.finite(eig) & eig > 0]
}

pick_min_node_size_oob_cpp <- function(target_mat, Z_pre,
                                       min_node_candidates = c(10L, 15L, 20L, 30L, 50L, 100L),
                                       learning_rate = 1,
                                       maxdepth = 100,
                                       subsample = 1,
                                       seed = NULL,
                                       tune_cols = NULL,
                                       mtry = NULL,
                                       min_bucket = 1L) {
  .check_cpp_cit_loaded_min()
  
  target_mat <- matrix2_double(target_mat)
  Z_pre <- matrix2_double(Z_pre)
  
  min_node_candidates <- as.integer(unique(min_node_candidates))
  min_node_candidates <- sort(min_node_candidates[!is.na(min_node_candidates) & min_node_candidates >= 1L])
  
  if (length(min_node_candidates) == 0L) {
    stop("No valid min_node_size candidates.")
  }
  
  if (!is.null(tune_cols)) {
    tune_cols <- as.integer(tune_cols)
    tune_cols <- tune_cols[tune_cols >= 1L & tune_cols <= ncol(target_mat)]
    
    if (length(tune_cols) == 0L) {
      stop("No valid tune_cols.")
    }
    
    target_for_tuning <- target_mat[, tune_cols, drop = FALSE]
  } else {
    target_for_tuning <- target_mat
  }
  
  score_mat <- cpp_cit_oob_score_min_node_matrix(
    X = Z_pre,
    Y = target_for_tuning,
    min_node_candidates = min_node_candidates,
    learning_rate = learning_rate,
    subsample = subsample,
    max_depth = as.integer(maxdepth),
    min_bucket = as.integer(min_bucket),
    mtry = if (is.null(mtry)) 0L else as.integer(mtry),
    seed_base = if (is.null(seed)) NA_integer_ else as.integer(seed)
  )
  
  rownames(score_mat) <- as.character(min_node_candidates)
  
  scores <- apply(score_mat, 1L, function(v) {
    v <- v[is.finite(v)]
    if (length(v) == 0L) return(Inf)
    mean(v)
  })
  
  score_se <- apply(score_mat, 1L, function(v) {
    v <- v[is.finite(v)]
    if (length(v) <= 1L) return(0)
    stats::sd(v) / sqrt(length(v))
  })
  
  if (all(!is.finite(scores))) {
    stop("All OOB errors were non-finite. Check sample size and subsample.")
  }
  
  finite_scores <- scores
  finite_scores[!is.finite(finite_scores)] <- Inf
  
  min_idx <- which.min(finite_scores)
  threshold <- scores[min_idx] + score_se[min_idx]
  
  eligible <- which(is.finite(scores) & scores <= threshold)
  best_min_node_size <- min(min_node_candidates[eligible])
  
  list(
    best_min_node_size = as.integer(best_min_node_size),
    scores = scores,
    score_se = score_se,
    score_mat = score_mat
  )
}

BLITZ <- function(x, y, z = NULL,
                                     learning_rate = 1,
                                     maxdepth = 100,
                                     min_node_size = c(5L, 10L, 15L, 20L, 40L, 80L),
                                     min_bucket = 1L,
                                     mtry = NULL,
                                     seed = NULL,
                                     return_tuning = FALSE,
                                     tune_cols_x = NULL,
                                     tune_cols_y = NULL,
                                     poly_degree = 2L,
                                     poly_interactions = TRUE,
                                     poly_ridge = 1e-8,
                                     subsample = 1) {
  .check_cpp_cit_loaded_min()
  
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
        tune_cols = tune_cols_x,
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
        tune_cols = tune_cols_y,
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
