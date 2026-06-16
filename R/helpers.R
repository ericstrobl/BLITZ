matrix2 <- function(a) {
  if (is.null(dim(a))) matrix(a, ncol = 1L) else as.matrix(a)
}

matrix2_double <- function(a) {
  a <- matrix2(a)
  if (!is.double(a)) storage.mode(a) <- "double"
  a
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
                                       mtry = NULL,
                                       min_bucket = 1L) {
  
  target_mat <- matrix2_double(target_mat)
  Z_pre <- matrix2_double(Z_pre)
  
  min_node_candidates <- as.integer(unique(min_node_candidates))
  min_node_candidates <- sort(min_node_candidates[!is.na(min_node_candidates) & min_node_candidates >= 1L])
  
  if (length(min_node_candidates) == 0L) {
    stop("No valid min_node_size candidates.")
  }
  
  score_mat <- cpp_cit_oob_score_min_node_matrix(
    X = Z_pre,
    Y = target_mat,
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
