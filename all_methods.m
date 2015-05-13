function priors_MSEs = all_methods(X_train, y_train, X_test, y_test, sigma, lambda)
    classes = sort(unique(y_test));
    true_prior = sum(y_test==classes(2))/length(y_test);
    [prior1, ~] = computePE_DR(X_train, y_train, X_test, sigma, lambda);
    MSE1 = computeMSE(prior1(2,2), true_prior);
    prior2 = competitors(X_train, y_train, X_test);
    MSE2 = computeMSE(prior2, true_prior);
    priors = [prior1(2,2), prior2(:)'];
    MSEs = [MSE1, MSE2(:)'];
    priors_MSEs = [priors, MSEs];
end