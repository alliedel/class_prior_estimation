clear, clc, close all

m = 1;  % number of features
sigma = 0.2;

[X_train, y_train, X_test, y_test] = getSyntheticData();

estimator = PE_DR(X_train, y_train);
[sigma, lambda] = tunePE_DR(X, y);
[priors, alphas] = computePE_DR(X_train, y_train, X_test, sigma, lambda);
[sigma, lambda] = tunePE_DR(X, y);
disp('Class priors are:')
disp(priors)

fprintf('end: ');

% scatter(X_train, 0.5*ones(size(X_train)), 50*ones(size(X_train)), y_train+1, 'x', 'LineWidth',1.3); hold on
% scatter(X_test, 1.5*ones(size(X_test)), 50*ones(size(X_test)), y_test+1, 'x', 'LineWidth',1.3); hold on
% axis([-0.2 1.2 0.45 1.55])

x = linspace(0,1,10000);
density_ratios = NaN(size(x));
for i = 1:length(x)
    phi_bold = evaluateBasis(x(i), X_train, estimator.sigma);
    density_ratios(i) = phi_bold'*alphas;
end
plot(x, density_ratios, 'g')

