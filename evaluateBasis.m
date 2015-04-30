function phi_bold = evaluateBasis(x, X_train, sigma)
% Evaluate the sample x on all the basis functions
    [n1, m1] = size(x);
    [~, m2] = size(X_train);
    if n1 ~= 1
        error('x must be a row vector')
    elseif m1 ~= m2
        error('Number of features do not match')
    end
    x_diff = bsxfun(@plus, x, -X_train);
    x_norm = sum(x_diff.^2,2);
    phi_bold = [1; exp(-x_norm/(2*sigma^2))];
end