function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Regularized linear regression cost function
matrix = X * theta - y;
matrix = matrix .^ 2;
J = sum(sum(matrix))/2/m;
         temp_matrix = theta;
         temp_matrix(1)=0;
         temp = sum(temp_matrix .^ 2);
         J += temp*lambda/2/m;

% Regularized linear regression gradient
matrix = X * theta - y;
grad(1) = sum(matrix .* X(:,1))/m;
for i=2:size(grad,1),
grad(i) = sum(matrix .* X(:,i))/m + lambda * theta(i)/m;
end;




% =========================================================================

grad = grad(:);

end
