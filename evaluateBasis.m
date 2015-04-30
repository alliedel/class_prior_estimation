function phi_bold_mat = evaluateBasis(X, X_train, sigma)
% Evaluate the sample x on all the basis functions
    [n1, m1] = size(X);
    [~, m2] = size(X_train);
    if m1 ~= m2
        error('Number of features do not match')
    end
    D = pdist2(X,X_train);
    phi_bold_mat = [ones(size(D,1),1) exp(-D/(2*sigma^2))];
    
end
