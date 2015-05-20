# class_prior_estimation
Implementation of a handful of class prior estimation methods.

+ [PE_DR] <-- duPlessis2013, "Semi-supervised learning of class balance under class-prior change by distribution matching"
+ [CC, ACC, Max, X, T50, MS, MM] <-- Forman2005, "Counting positives accurately despite inaccurate classification"
+ [PA, SPA, SCC] <-- Bella2010, "Quantification via probability estimators"
+ [EM] <-- Saerens2002, "Adjusting the outputs of a classifier to new a priori probabilities: a simple procedure"

## Basic usage
The methods are split into three functions based on their approach. 

1. `[priors, alphas] = computePE_DR(X_train, y_train, X_test, sigma, lambda);` where priors is an 2xc array of [classes; priors]. sigma is a kernel bandwidth parameter and lambda is a regularization constant
2. `[priors] = classification_methods(X_train, y_train, X_test);`where priors is a 1x11 vector of positive class priors for all the other methods, listed in the order [CC,ACC,Max,X,T50,MS,MM,PA,SPA,SCC,EM].
3. `[prior] = oracle(X_test, pos_pdf, neg_pdf);` where the PDFs are functional handles for the positive and negative classes, respectively. Uses a maximum likelihood estimate with Newton's method optimization to estimate the prior -- this is a best case scenario.

All 13 methods can be called in a single shot with `all_methods(X_train, y_train, X_test, sigma, lambda)`.

## Other functionality
bootstrap_traintest -- simultaneously bootstraps training and testing data and computes several confidence interval metrics
replot -- plots the results from example.m

## Example ##
An example of all the methods over multiple datasets and a range of test class priors can be found in `example.m`. Plots of some results of this script are available in *results/* as .eps figures and .mat data files

## Known Issues ##
Logistic regression is used for the classifier scores in classification_methods. If the data is linearly separable (i.e. if the number of samples is not much larger than the number of dimensions), MATLAB's logreg will either return NaN weights, error out, or create CDF issues if the probabilities are all extreme. Attempts at fixing this, with mild success, have included PCA to reduce dimensionality or using L1 regularization.


