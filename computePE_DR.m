function [ priors ] = computePE_DR( X1, y1, X2, sigma, lambda )
%PE_DR Class prior estimation using Pearson divergence with density ratio
%estimation [duPlessis2013]
%   Inputs:
%       X1 - n x m matrix of samples, where n is the number of
%       samples and m is the number of features
%       y1 - n x 1 vector of training class labels
%       X2 - n' x m matrix of testing samples
%       sigma - Kernel bandwidth
%       lambda - regularization constant
%   Output:
%       priors - c x 2 vector of estimated class priors, where c is the 
%       number of classes, column 1 is the class label and column 2 is the
%       estimated prior
    
    classes = sort(unique(y1));
    R = eye(size(X1, 1)+1);
    R(1, 1) = 0;  % do not regularize constant basis function
    G = computeG(X1, X2, sigma);
    H = computeH(X1, y1, sigma, classes);
    GR_inv = inv(G+lambda*R);
    %% CVX Optimization
    c = length(classes);
    cvx_begin
        variable theta(c)
        minimize( -0.5*theta'*H'*GR_inv*G*GR_inv*H*theta + theta'*H'*GR_inv'*H*theta-0.5 )
        subject to
            ones(size(classes))' * theta == 1
    cvx_end
    priors = theta;
end

function G_hat = computeG(X_train, X_test, sigma)
% Compute G_hat
    n_prime = size(X_test, 1);
    G_hat = zeros(size(X_train, 1)+1);
    for i = 1:n_prime
        x = X_test(i, :);
        phi_bold = evaluateBasis(x, X_train, sigma);
        G_hat = G_hat + phi_bold*phi_bold';
    end
    G_hat = G_hat./n_prime;   
end

function H_hat = computeH(X_train, y_train, sigma, classes)
% Compute H_hat
    H_hat = zeros(length(y_train)+1, length(classes));
    n_y = zeros(length(classes), 1);  % number of samples from each class
    for i = 1:length(y_train)
        yi = y_train(i);
        x = X_train(i,:);
        phi_bold = evaluateBasis(x, X_train, sigma);
        class_index = find(classes-yi, 1);
        H_hat(:, class_index) = H_hat(:, class_index) + phi_bold;
        n_y(class_index) = n_y(class_index)+1;
    end
    H_hat = bsxfun(@times, H_hat, 1./n_y');
end

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
    x_norm = sqrt(sum(x_diff.^2,2));
    phi_bold = [1; exp(-x_norm/(2*sigma^2))];
end

