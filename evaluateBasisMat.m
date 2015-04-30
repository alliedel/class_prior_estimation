function phi_bold_mat = evaluateBasisMat(X, X_train, sigma)
% Evaluate the sample x on all the basis functions
    [n1, m1] = size(X);
    [~, m2] = size(X_train);
    if m1 ~= m2
        error('Number of features do not match')
    end
    D = pdist2(X,X_train);
%     x_diff = bsxfun(@plus, x, -X_train);
%     x_norm = sum(x_diff.^2,2,'euclidean');
    phi_bold_mat = exp(-D/(2*sigma^2));
%     phi_bold = [1; exp(-x_norm/(2*sigma^2))];
end
