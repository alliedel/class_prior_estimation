function [ prior ] = oracle( X_test, pos_pdf, neg_pdf )
%ORACLE Given the true underlying distributions, using MLE to estimate
%class priors. This is the best case scenario
%   Inputs:
%       X_test - test samples, mxn (m #samples, n is #features)
%       pos_pdf - function handle for positive class pdf
%       neg_pdf - function handle for negative class pdf
%
%   Outputs:
%       prior - estimated positive class prior
disp('Solving oracle...')
p_pos = pos_pdf(X_test);
p_neg = neg_pdf(X_test);
% Solve w/ Newton's method
fun = @(x) sum(log((p_pos-p_neg)*x + p_neg));
gradient = @(x) sum(1./(x+p_neg./(p_pos-p_neg)));
hessian = @(x) -sum((p_pos-p_neg).^2./((p_pos-p_neg)*x+p_neg).^2);
e_stop = 1E-12;
max_iter = 100;
iter = 0;
prior = 0.5;
f_new = fun(prior);
f_old = Inf;
while abs(f_new - f_old) > e_stop && iter<max_iter
    disp(sprintf('     Newtons method objective %0.5g', f_new))
    prior = prior - 1/hessian(prior)*gradient(prior);
    iter = iter+1;
    f_old = f_new;
    f_new = fun(prior);
end  
disp('Solved')

end

