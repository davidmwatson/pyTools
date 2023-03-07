#!/usr/bin/env Rscript

## Helper functions for calculating Cohen's d and Hedges' g for paired
## and indepednent designs, with choice of standardizer. Sticking it here
## because I can never remember how to do these otherwise.


calc.cohens.d <- function(x, y, paired, standardizer="av") {
  ## Returns Cohen's d for paired or unpaired comparisons
  ## standardizer can be one of "z", "av", or "rm" (N/A if paired=F)

  if ( paired ) {  # within-subjects
    m <- mean(x-y)
    if (standardizer == "z") {
      s <- sd(x-y)
    } else if (standardizer == "av") {
      s <- (sd(x) + sd(y)) / 2
    } else if (standardizer == "rm") {
      s <- sd(x-y) * sqrt(2 * (1 - cor(x,y)))
    } else {
      stop("Invalid standardizer")
    }

    } else {  # between-subjects
      m <- mean(x) - mean(y)
      nx <- length(x)
      ny <- length(y)
      s <- sqrt( ((nx-1)*var(x) + (ny-1)*var(y)) / (nx + ny - 2) )
    }

  return(m/s)
}


calc.hedges.g <- function(x, y, paired, ...) {
  ## Calculate Hedges' g for paired or unpaired comparisons
  d <- calc.cohens.d(x, y, paired, ...)
  if ( paired ) {  # within-subjects
    dof <- length(x) - 1
    J <- 1 - ( 3 / (4 * dof - 1) )
  } else {  # between-subjects
    J <- 1 - ( 3 / (4 * (length(x) + length(y)) - 9) )
  }
  return(d * J)
}
