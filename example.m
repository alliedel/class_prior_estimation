clear, clc, close all

m = 1;  % number of features
sigma = 0.2;

X_train = [0.25 + sigma*randn(1000,1);
           rand(500,1);
           0.75 + sigma*randn(1000,1);
           rand(500,1)];  % 50/50 split
y_train = [ones(1500,1);
           zeros(1500,1)];  % 50/50 split
X_test = [0.25 + sigma*randn(500,1);
           rand(250,1);
           0.75 + sigma*randn(1500,1);
           rand(750,1)];  % 0.25
y_test = [ones(750,1);
           zeros(2250,1)];  % 0.25

scatter(X_train, 0.5*ones(size(X_train)), 50*ones(size(X_train)), y_train+1, 'x', 'LineWidth',1.3); hold on
scatter(X_test, 1.5*ones(size(X_test)), 50*ones(size(X_test)), y_test+1, 'x', 'LineWidth',1.3); hold on
axis([-0.2 1.2 0.45 1.55])


estimator = PE_DR(X_train, y_train);
[priors, alphas] = estimator.estimateClassBalance(X_test);

x = linspace(0,1,10000);
density_ratios = NaN(size(x));
for i = 1:length(x)
    phi_bold = evaluateBasis(x(i), X_train, estimator.sigma);
    density_ratios(i) = phi_bold'*alphas;
end
plot(x, density_ratios, 'g')


