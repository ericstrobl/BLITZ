# BLITZ

BLITZ (Broad-to-Local Independence Testing via residualiZation) is a fast nonparametric conditional independence test for causal discovery. It uses two-stage residualization: a low-order polynomial regression first removes broad smooth dependence on the conditioning set, and shallow tree regressions then remove remaining nonlinear conditional structure from compact transformed residual features. Conditional independence is tested using the residual cross-covariance statistic with a moment-matched chi-square null approximation. The academic article describing BLITZ in detail can be found [here](https://arxiv.org/abs/2606.18011).

BLITZ is designed for repeated scalar conditional independence queries of the form X ⫫ Y | Z, as required by constraint-based causal discovery algorithms such as PC, FCI, and RFCI.

All code was tested in R version 4.3.1.

# Installation

BLITZ uses Rcpp/C++, so a working C++ toolchain is required.

On Windows, install Rtools.

On macOS, install Xcode Command Line Tools.

On Linux, install `g++` or the standard build tools for your distribution.

> install.packages("remotes")

> remotes::install_github("ericstrobl/BLITZ")

> library(BLITZ)

# Sample Run

> n <- 1000

> z <- matrix(rnorm(n * 2), nrow = n, ncol = 2)

> x <- z[, 1] + z[, 2]^2 + rnorm(n)

> y <- sin(z[, 1]) + z[, 2] + rnorm(n)

> fit <- BLITZ(x, y, z)

> fit$p

> fit$Sta
