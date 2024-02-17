#!/usr/bin/env Rscript

## Helper functions for calculating Cohen's d and Hedges' g for paired
## and indepednent designs, with choice of standardizer. Sticking it here
## because I can never remember how to do these otherwise.


calc.cohens.d <- function(x, y=NULL, paired=F, standardizer="av") {
    ## Cohen's d for independent, paired, or one-sample comparisons.
    ## standardizer can be one of "z", "av", or "rm" (only matters for
    ## paired samples).

    if ( is.null(y) ) {  # one-sample
        m <- mean(x)
        s <- sd(x)

    } else if ( paired ) {  # within-subjects
        m <- mean(x-y)
        if ( standardizer == "z" ) {
            s <- sd(x-y)
        } else if ( standardizer == "av" ) {
            s <- (sd(x) + sd(y)) / 2
        } else if ( standardizer == "rm" ) {
            r <- cor(x,y)
            s <- sqrt(var(x) + var(y) - 2 * r * sd(x) * sd(y)) / sqrt(2 * (1 - r))
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


calc.hedges.g <- function(x, y=NULL, paired=F, ...) {
    ## Hedges' g for independent, paired, or one-sample comparisons.
    ## Arguments as per calc.cohens.d function.
    d <- calc.cohens.d(x, y, paired, ...)
    if ( is.null(y) || paired ) {  # one-sample or within-subjects
        J <- 1 - (3 / (4 * (length(x) - 1) - 1))
    } else {  # between-subjects
        J <- 1 - (3 / (4 * (length(x) + length(y)) - 9))
    }
    return(d * J)
}
