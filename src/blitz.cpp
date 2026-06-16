#include <RcppArmadillo.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

struct Node {
  bool leaf;
  int feature;
  double split;
  double value;
  int left;
  int right;
};

static void set_seed_if_needed(int seed) {
  if (seed != NA_INTEGER) {
    Function set_seed("set.seed");
    set_seed(seed);
  }
}

static int draw_int_0_to_nminus1(int n) {
  int out = static_cast<int>(std::floor(R::unif_rand() * n));
  if (out >= n) out = n - 1;
  if (out < 0) out = 0;
  return out;
}

static std::vector<int> sample_features_cpp(int p, int mtry) {
  std::vector<int> features(p);
  for (int j = 0; j < p; ++j) features[j] = j;
  
  if (mtry < 1 || mtry >= p) {
    return features;
  }
  
  for (int k = 0; k < mtry; ++k) {
    int j = k + draw_int_0_to_nminus1(p - k);
    std::swap(features[k], features[j]);
  }
  
  features.resize(mtry);
  return features;
}

static bool row_less_feature_cpp(const NumericMatrix& X,
                                 int feature,
                                 int a,
                                 int b) {
  double xa = X(a, feature);
  double xb = X(b, feature);
  
  if (xa < xb) return true;
  if (xa > xb) return false;
  return a < b;
}

static std::vector< std::vector<int> > root_sorted_rows_cpp(
    const NumericMatrix& X,
    const std::vector<int>& weights) {
  int n = X.nrow();
  int p = X.ncol();
  
  std::vector<int> active_rows;
  active_rows.reserve(n);
  
  for (int i = 0; i < n; ++i) {
    if (weights[i] > 0) {
      active_rows.push_back(i);
    }
  }
  
  if (active_rows.empty()) {
    stop("No active training rows.");
  }
  
  std::vector< std::vector<int> > sorted_by_feature(p);
  
  for (int j = 0; j < p; ++j) {
    sorted_by_feature[j] = active_rows;
    
    std::sort(
      sorted_by_feature[j].begin(),
      sorted_by_feature[j].end(),
      [&](int a, int b) {
        return row_less_feature_cpp(X, j, a, b);
      }
    );
  }
  
  return sorted_by_feature;
}

static void evaluate_presorted_feature_cpp(
    const NumericMatrix& X,
    const NumericVector& y,
    const std::vector<int>& weights,
    const std::vector<int>& sorted_rows,
    int feature,
    double total_w,
    double sum_y,
    double sum_y2,
    int min_bucket,
    double& best_sse,
    int& best_feature,
    double& best_split) {
  
  const int m = sorted_rows.size();
  
  double left_w = 0.0;
  double left_sum = 0.0;
  double left_sum2 = 0.0;
  
  for (int pos = 0; pos < m - 1; ++pos) {
    int r = sorted_rows[pos];
    double w = static_cast<double>(weights[r]);
    double yy = y[r];
    
    left_w += w;
    left_sum += w * yy;
    left_sum2 += w * yy * yy;
    
    double right_w = total_w - left_w;
    
    if (left_w < min_bucket) continue;
    if (right_w < min_bucket) break;
    
    double x_left = X(r, feature);
    double x_right = X(sorted_rows[pos + 1], feature);
    
    if (x_left == x_right) continue;
    
    double right_sum = sum_y - left_sum;
    double right_sum2 = sum_y2 - left_sum2;
    
    double left_sse = left_sum2 - (left_sum * left_sum) / left_w;
    double right_sse = right_sum2 - (right_sum * right_sum) / right_w;
    double sse = left_sse + right_sse;
    
    if (R_finite(sse) && sse < best_sse) {
      best_sse = sse;
      best_feature = feature;
      best_split = x_left + 0.5 * (x_right - x_left);
    }
  }
}

static int build_tree_presorted_cpp(
    const NumericMatrix& X,
    const NumericVector& y,
    const std::vector<int>& weights,
    const std::vector< std::vector<int> >& sorted_by_feature,
    int depth,
    int max_depth,
    int min_node_size,
    int min_bucket,
    int mtry,
    std::vector<Node>& nodes) {
  
  const int p = X.ncol();
  const std::vector<int>& rows0 = sorted_by_feature[0];
  const int n_unique = rows0.size();
  
  double total_w = 0.0;
  double sum_y = 0.0;
  double sum_y2 = 0.0;
  
  for (int idx = 0; idx < n_unique; ++idx) {
    int r = rows0[idx];
    double w = static_cast<double>(weights[r]);
    double yy = y[r];
    
    total_w += w;
    sum_y += w * yy;
    sum_y2 += w * yy * yy;
  }
  
  Node node;
  node.leaf = true;
  node.feature = -1;
  node.split = NA_REAL;
  node.value = sum_y / total_w;
  node.left = -1;
  node.right = -1;
  
  int node_id = nodes.size();
  nodes.push_back(node);
  
  // min_node_size is weighted parent-node size.
  // min_bucket is weighted minimum child size.
  if (total_w < min_node_size ||
      total_w < 2.0 * min_bucket ||
      depth >= max_depth ||
      n_unique <= 1) {
    return node_id;
  }
  
  double parent_sse = sum_y2 - (sum_y * sum_y) / total_w;
  if (!(parent_sse > 0.0) || !R_finite(parent_sse)) {
    return node_id;
  }
  
  double best_sse = std::numeric_limits<double>::infinity();
  int best_feature = -1;
  double best_split = NA_REAL;
  
  if (mtry >= p) {
    for (int j = 0; j < p; ++j) {
      evaluate_presorted_feature_cpp(
        X,
        y,
        weights,
        sorted_by_feature[j],
        j,
        total_w,
        sum_y,
        sum_y2,
        min_bucket,
        best_sse,
        best_feature,
        best_split
      );
    }
  } else {
    std::vector<int> features = sample_features_cpp(p, mtry);
    
    for (int ff = 0; ff < static_cast<int>(features.size()); ++ff) {
      int j = features[ff];
      
      evaluate_presorted_feature_cpp(
        X,
        y,
        weights,
        sorted_by_feature[j],
        j,
        total_w,
        sum_y,
        sum_y2,
        min_bucket,
        best_sse,
        best_feature,
        best_split
      );
    }
  }
  
  double tol = 1e-12 * (std::fabs(parent_sse) + 1.0);
  if (best_feature < 0 || !(best_sse < parent_sse - tol)) {
    return node_id;
  }
  
  std::vector< std::vector<int> > left_sorted(p);
  std::vector< std::vector<int> > right_sorted(p);
  
  double left_w_check = 0.0;
  double right_w_check = 0.0;
  
  for (int j = 0; j < p; ++j) {
    left_sorted[j].reserve(sorted_by_feature[j].size());
    right_sorted[j].reserve(sorted_by_feature[j].size());
    
    for (int idx = 0; idx < static_cast<int>(sorted_by_feature[j].size()); ++idx) {
      int r = sorted_by_feature[j][idx];
      
      if (X(r, best_feature) <= best_split) {
        left_sorted[j].push_back(r);
      } else {
        right_sorted[j].push_back(r);
      }
    }
  }
  
  for (int idx = 0; idx < static_cast<int>(left_sorted[0].size()); ++idx) {
    left_w_check += static_cast<double>(weights[left_sorted[0][idx]]);
  }
  
  for (int idx = 0; idx < static_cast<int>(right_sorted[0].size()); ++idx) {
    right_w_check += static_cast<double>(weights[right_sorted[0][idx]]);
  }
  
  if (left_w_check < min_bucket || right_w_check < min_bucket) {
    return node_id;
  }
  
  nodes[node_id].leaf = false;
  nodes[node_id].feature = best_feature;
  nodes[node_id].split = best_split;
  
  int left_id = build_tree_presorted_cpp(
    X,
    y,
    weights,
    left_sorted,
    depth + 1,
    max_depth,
    min_node_size,
    min_bucket,
    mtry,
    nodes
  );
  
  int right_id = build_tree_presorted_cpp(
    X,
    y,
    weights,
    right_sorted,
    depth + 1,
    max_depth,
    min_node_size,
    min_bucket,
    mtry,
    nodes
  );
  
  nodes[node_id].left = left_id;
  nodes[node_id].right = right_id;
  
  return node_id;
}

static double predict_one_cpp(const NumericMatrix& X,
                              const std::vector<Node>& nodes,
                              int row) {
  int id = 0;
  
  while (!nodes[id].leaf) {
    const Node& nd = nodes[id];
    
    if (X(row, nd.feature) <= nd.split) {
      id = nd.left;
    } else {
      id = nd.right;
    }
  }
  
  return nodes[id].value;
}

static std::vector<Node> fit_tree_weighted_presorted_cpp(
    const NumericMatrix& X,
    const NumericVector& y,
    const std::vector<int>& weights,
    int max_depth,
    int min_node_size,
    int min_bucket,
    int mtry) {
  
  std::vector< std::vector<int> > sorted_by_feature =
    root_sorted_rows_cpp(X, weights);
  
  int active_count = sorted_by_feature[0].size();
  
  std::vector<Node> nodes;
  nodes.reserve(2 * active_count + 1);
  
  build_tree_presorted_cpp(
    X,
    y,
    weights,
    sorted_by_feature,
    0,
    max_depth,
    min_node_size,
    min_bucket,
    mtry,
    nodes
  );
  
  return nodes;
}

static NumericVector fit_predict_tree_weighted_all_cpp(
    const NumericMatrix& X,
    const NumericVector& y,
    const std::vector<int>& weights,
    int max_depth,
    int min_node_size,
    int min_bucket,
    int mtry) {
  
  int n = X.nrow();
  
  std::vector<Node> nodes = fit_tree_weighted_presorted_cpp(
    X,
    y,
    weights,
    max_depth,
    min_node_size,
    min_bucket,
    mtry
  );
  
  NumericVector pred(n);
  
  for (int i = 0; i < n; ++i) {
    pred[i] = predict_one_cpp(X, nodes, i);
  }
  
  return pred;
}

static NumericVector fit_predict_tree_weighted_rows_cpp(
    const NumericMatrix& X,
    const NumericVector& y,
    const std::vector<int>& weights,
    const std::vector<int>& pred_rows,
    int max_depth,
    int min_node_size,
    int min_bucket,
    int mtry) {
  
  std::vector<Node> nodes = fit_tree_weighted_presorted_cpp(
    X,
    y,
    weights,
    max_depth,
    min_node_size,
    min_bucket,
    mtry
  );
  
  NumericVector pred(pred_rows.size());
  
  for (int i = 0; i < static_cast<int>(pred_rows.size()); ++i) {
    pred[i] = predict_one_cpp(X, nodes, pred_rows[i]);
  }
  
  return pred;
}

static std::vector<int> bootstrap_weights_cpp(int n,
                                              double subsample,
                                              std::vector<int>& inbag) {
  if (!R_finite(subsample) || subsample <= 0.0 || subsample > 1.0) {
    stop("subsample must be in (0, 1].");
  }
  
  int n_boot = std::max(1, static_cast<int>(std::floor(subsample * n)));
  
  std::fill(inbag.begin(), inbag.end(), 0);
  
  for (int i = 0; i < n_boot; ++i) {
    int r = draw_int_0_to_nminus1(n);
    ++inbag[r];
  }
  
  return inbag;
}

static std::vector<int> unit_weights_cpp(int n) {
  std::vector<int> weights(n, 1);
  return weights;
}

static void check_X_cpp(const NumericMatrix& X) {
  int n = X.nrow();
  int p = X.ncol();
  
  if (p < 1) {
    stop("X must have at least one column.");
  }
  
  for (int j = 0; j < p; ++j) {
    for (int i = 0; i < n; ++i) {
      if (!R_finite(X(i, j))) {
        stop("X contains non-finite values.");
      }
    }
  }
}

static void check_Y_cpp(const NumericMatrix& Y) {
  int n = Y.nrow();
  int q = Y.ncol();
  
  if (q < 1) {
    stop("Y must have at least one column.");
  }
  
  for (int j = 0; j < q; ++j) {
    for (int i = 0; i < n; ++i) {
      if (!R_finite(Y(i, j))) {
        stop("Y contains non-finite values.");
      }
    }
  }
}

static void normalize_tree_params_cpp(const NumericMatrix& X,
                                      int& max_depth,
                                      int& min_bucket,
                                      int& mtry) {
  int p = X.ncol();
  
  if (max_depth <= 0) {
    max_depth = std::numeric_limits<int>::max() / 4;
  }
  
  if (min_bucket < 1) {
    stop("min_bucket must be positive.");
  }
  
  if (mtry <= 0) {
    mtry = std::max(1, static_cast<int>(std::floor(std::sqrt(static_cast<double>(p)))));
  }
  
  if (mtry > p) {
    mtry = p;
  }
}

static int poly_feature_count_cpp(int p,
                                  int degree,
                                  bool interactions) {
  if (degree < 0 || degree > 3) {
    stop("poly_degree must be 0, 1, 2, or 3.");
  }
  
  int d = 1; // intercept
  
  if (degree >= 1) {
    d += p;
  }
  
  if (degree >= 2) {
    d += p;
    
    if (interactions) {
      d += p * (p - 1) / 2;
    }
  }
  
  if (degree >= 3) {
    d += p;
    
    if (interactions) {
      d += p * (p - 1);
      d += p * (p - 1) * (p - 2) / 6;
    }
  }
  
  return d;
}

static void fill_poly_row_cpp(const NumericMatrix& Z,
                              int row,
                              int degree,
                              bool interactions,
                              std::vector<double>& phi) {
  int p = Z.ncol();
  int idx = 0;
  
  phi[idx++] = 1.0;
  
  if (degree >= 1) {
    for (int j = 0; j < p; ++j) {
      phi[idx++] = Z(row, j);
    }
  }
  
  if (degree >= 2) {
    for (int j = 0; j < p; ++j) {
      double zj = Z(row, j);
      phi[idx++] = zj * zj;
    }
    
    if (interactions) {
      for (int j = 0; j < p - 1; ++j) {
        for (int k = j + 1; k < p; ++k) {
          phi[idx++] = Z(row, j) * Z(row, k);
        }
      }
    }
  }
  
  if (degree >= 3) {
    for (int j = 0; j < p; ++j) {
      double zj = Z(row, j);
      phi[idx++] = zj * zj * zj;
    }
    
    if (interactions) {
      for (int j = 0; j < p - 1; ++j) {
        double zj = Z(row, j);
        
        for (int k = j + 1; k < p; ++k) {
          double zk = Z(row, k);
          phi[idx++] = zj * zj * zk;
          phi[idx++] = zj * zk * zk;
        }
      }
      
      for (int j = 0; j < p - 2; ++j) {
        for (int k = j + 1; k < p - 1; ++k) {
          for (int l = k + 1; l < p; ++l) {
            phi[idx++] = Z(row, j) * Z(row, k) * Z(row, l);
          }
        }
      }
    }
  }
}

// [[Rcpp::export]]
NumericMatrix cpp_poly_residualize_matrix(NumericMatrix Z,
                                          NumericMatrix A,
                                          int degree = 2,
                                          bool interactions = true,
                                          double ridge = 1e-8) {
  int n = Z.nrow();
  int p = Z.ncol();
  int q = A.ncol();
  
  if (A.nrow() != n) {
    stop("A and Z must have the same number of rows.");
  }
  
  if (p < 1) {
    stop("Z must have at least one column.");
  }
  
  if (q < 1) {
    stop("A must have at least one column.");
  }
  
  check_X_cpp(Z);
  check_Y_cpp(A);
  
  int d = poly_feature_count_cpp(p, degree, interactions);
  
  arma::mat XtX(d, d, arma::fill::zeros);
  arma::mat XtY(d, q, arma::fill::zeros);
  
  std::vector<double> phi(d);
  
  for (int i = 0; i < n; ++i) {
    fill_poly_row_cpp(Z, i, degree, interactions, phi);
    
    for (int a = 0; a < d; ++a) {
      for (int j = 0; j < q; ++j) {
        XtY(a, j) += phi[a] * A(i, j);
      }
      
      for (int b = 0; b <= a; ++b) {
        XtX(a, b) += phi[a] * phi[b];
      }
    }
  }
  
  for (int a = 0; a < d; ++a) {
    for (int b = 0; b < a; ++b) {
      XtX(b, a) = XtX(a, b);
    }
  }
  
  if (ridge > 0.0) {
    for (int a = 1; a < d; ++a) {
      XtX(a, a) += ridge;
    }
  }
  
  arma::mat XtX_ridge = XtX;
  
  double ridge_eff = std::max(ridge, 1e-8) * static_cast<double>(n);
  
  for (int a = 1; a < d; ++a) {
    XtX_ridge(a, a) += ridge_eff;
  }
  
  arma::mat Beta;
  bool ok = arma::solve(
    Beta,
    XtX_ridge,
    XtY,
    arma::solve_opts::likely_sympd + arma::solve_opts::no_approx
  );
  
  if (!ok) {
    ok = arma::solve(
      Beta,
      XtX_ridge,
      XtY,
      arma::solve_opts::no_approx
    );
  }
  
  if (!ok) {
    Beta = arma::pinv(XtX_ridge) * XtY;
  }
  
  NumericMatrix out(n, q);
  
  for (int i = 0; i < n; ++i) {
    fill_poly_row_cpp(Z, i, degree, interactions, phi);
    
    for (int j = 0; j < q; ++j) {
      double fitted = 0.0;
      
      for (int a = 0; a < d; ++a) {
        fitted += phi[a] * Beta(a, j);
      }
      
      out(i, j) = A(i, j) - fitted;
    }
  }
  
  return out;
}

// [[Rcpp::export]]
NumericMatrix cpp_cit_oob_score_min_node_matrix(NumericMatrix X,
                                                NumericMatrix Y,
                                                IntegerVector min_node_candidates,
                                                double learning_rate = 1.0,
                                                double subsample = 1.0,
                                                int max_depth = 100,
                                                int min_bucket = 1,
                                                int mtry = 0,
                                                int seed_base = NA_INTEGER) {
  RNGScope scope;
  
  int n = X.nrow();
  int q = Y.ncol();
  
  if (Y.nrow() != n) {
    stop("Y must have the same number of rows as X.");
  }
  
  check_X_cpp(X);
  check_Y_cpp(Y);
  
  std::vector<int> min_nodes;
  min_nodes.reserve(min_node_candidates.size());
  
  for (int i = 0; i < min_node_candidates.size(); ++i) {
    int v = min_node_candidates[i];
    
    if (v >= 1) {
      if (std::find(min_nodes.begin(), min_nodes.end(), v) == min_nodes.end()) {
        min_nodes.push_back(v);
      }
    }
  }
  
  std::sort(min_nodes.begin(), min_nodes.end());
  
  if (min_nodes.empty()) {
    stop("No valid min_node_size candidates.");
  }
  
  int max_depth_use = max_depth;
  int min_bucket_use = min_bucket;
  int mtry_use = mtry;
  
  normalize_tree_params_cpp(
    X,
    max_depth_use,
    min_bucket_use,
    mtry_use
  );
  
  int K = min_nodes.size();
  NumericMatrix out(K, q);
  NumericVector y(n);
  
  for (int j = 0; j < q; ++j) {
    for (int i = 0; i < n; ++i) {
      y[i] = Y(i, j);
    }
    
    int seed_target = NA_INTEGER;
    if (seed_base != NA_INTEGER) {
      seed_target = seed_base + (j + 1);
    }
    
    std::vector<int> inbag(n);
    
    if (seed_target != NA_INTEGER) {
      set_seed_if_needed(seed_target);
    }
    
    std::vector<int> weights = bootstrap_weights_cpp(n, subsample, inbag);
    
    std::vector<int> oob_rows;
    oob_rows.reserve(n);
    
    for (int i = 0; i < n; ++i) {
      if (weights[i] == 0) {
        oob_rows.push_back(i);
      }
    }
    
    if (oob_rows.empty()) {
      for (int k_idx = 0; k_idx < K; ++k_idx) {
        out(k_idx, j) = NA_REAL;
      }
      continue;
    }
    
    for (int k_idx = 0; k_idx < K; ++k_idx) {
      int k = min_nodes[k_idx];
      
      if (seed_target != NA_INTEGER) {
        // Replay bootstrap RNG so random-feature sampling starts from the
        // same point for each min_node_size candidate.
        std::vector<int> dummy_inbag(n);
        set_seed_if_needed(seed_target);
        bootstrap_weights_cpp(n, subsample, dummy_inbag);
      }
      
      NumericVector pred_oob = fit_predict_tree_weighted_rows_cpp(
        X,
        y,
        weights,
        oob_rows,
        max_depth_use,
        k,
        min_bucket_use,
        mtry_use
      );
      
      double sse = 0.0;
      
      for (int idx = 0; idx < static_cast<int>(oob_rows.size()); ++idx) {
        int i = oob_rows[idx];
        double e = y[i] - learning_rate * pred_oob[idx];
        sse += e * e;
      }
      
      out(k_idx, j) = sse / static_cast<double>(oob_rows.size());
    }
  }
  
  return out;
}

// [[Rcpp::export]]
NumericMatrix cpp_cit_residualize_matrix_one(NumericMatrix X,
                                             NumericMatrix Y,
                                             IntegerVector min_node_by_col,
                                             double learning_rate = 1.0,
                                             int max_depth = 100,
                                             int min_bucket = 1,
                                             int mtry = 0,
                                             int seed_base = NA_INTEGER) {
  RNGScope scope;
  
  int n = X.nrow();
  int q = Y.ncol();
  
  if (Y.nrow() != n) {
    stop("Y must have the same number of rows as X.");
  }
  
  check_X_cpp(X);
  check_Y_cpp(Y);
  
  int max_depth_use = max_depth;
  int min_bucket_use = min_bucket;
  int mtry_use = mtry;
  
  normalize_tree_params_cpp(
    X,
    max_depth_use,
    min_bucket_use,
    mtry_use
  );
  
  if (!(min_node_by_col.size() == 1 || min_node_by_col.size() == q)) {
    stop("min_node_by_col must have length 1 or ncol(Y).");
  }
  
  std::vector<int> weights = unit_weights_cpp(n);
  
  NumericMatrix out(n, q);
  NumericVector y(n);
  
  for (int j = 0; j < q; ++j) {
    int mn = min_node_by_col.size() == 1 ? min_node_by_col[0] : min_node_by_col[j];
    
    if (mn < 1) {
      stop("min_node_by_col must contain positive integers.");
    }
    
    for (int i = 0; i < n; ++i) {
      y[i] = Y(i, j);
    }
    
    if (seed_base != NA_INTEGER) {
      set_seed_if_needed(seed_base + (j + 1));
    }
    
    NumericVector pred = fit_predict_tree_weighted_all_cpp(
      X,
      y,
      weights,
      max_depth_use,
      mn,
      min_bucket_use,
      mtry_use
    );
    
    for (int i = 0; i < n; ++i) {
      out(i, j) = y[i] - learning_rate * pred[i];
    }
  }
  
  return out;
}
