clear, clc, close all

%m = 10;  % number of features
%X = [randn(100, m);randn(100,m)+1];
%y = [zeros(100, 1); ones(100,1)];
%X2 = [randn(50, m);randn(200,m)+1];
%y2 = [zeros(50, 1); ones(200,1)];
[X, y, X_test, y_test] = getSyntheticData();

estimator = PE_DR(X, y);
priors = estimator.estimateClassBalance(X_test);