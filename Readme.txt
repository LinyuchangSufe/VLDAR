Dataset
sector_index: The named 'sector_index' dataset contains weekly closing prices three S&P500 sector indices which are closely related to people's livelihood, i.e., consumer discretionary (CD), consumer staples (CS) and health care (HC). The dataset spans from July 27, 1998 to September 8, 2022, and is downloaded from the website of Yahoo Finance (\textit{https://finance.yahoo.com}).

Programs
rcpp_LDAR.cpp: Basic functions in form of Rcpp code.

LDAR.R: R functions to perform the proposed estimation and inference as well as to support 'rcpp_LDAR.cpp'.

KSD.R: R functions to conduct the Kernelized Stein Discrepancy (KSD) test.

DWQRE.R: R functions to calculate the doubly weighted quantile regression estimator (DWQRE).

Rcode_for_LDAR.html: This file contains all outputs for real data analysis of Liu et al. (2022+). Specifically, (1) Model estimations of the LDAR model based on EQMLE, GQMLE and DWQRE are presented in each subsection; (2) Based on the residuals of E-QMLE, we use Q-Q plot and KSD test to check the error distribution, and conduct portmanteau test to check the adequacy of fitted models; (3) The AREs are provided to compare the efficiency of three estimation methods. (4) Rolling forecasting and backtests are presented.

Rcode_for_LDAR.Rmd: Source codes for 'Rcode_for_LDAR_github.html'.