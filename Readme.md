Dataset
sector_index: The named 'SP500sector.csv' dataset contains weekly closing prices three S&P500 sector indices which are closely related to people's livelihood, i.e., consumer discretionary (CD), consumer staples (CS) and health care (HC). The dataset spans from July 27, 1998 to September 8, 2022, and is downloaded from the website of Yahoo Finance (https://finance.yahoo.com).

Programs
All programs are in mian folder,
  'SQMLE_VLDAR_BCD.py' is our proposed estimation method for vector linear double autoregressive model.
  'CCCGARCH2.py' is for estimating CCC-GARCH model.  
  'LDAR.py' is for estimating univariate linear double autoregressive model.
Note that all these methods are estimated by quasi-MLE.

real_data_analysis.html: This file contains all outputs for real data analysis of Lin and Zhu (2022+). Specifically, (1) Descriptive statistical analysis for the centered log-return of the proposed dataset; (2) Model estimations and inferences (BIC  and portmanteau test) of the VLDAR model based on QMLE; (3) Recursive forecasting and backtests are presented.
