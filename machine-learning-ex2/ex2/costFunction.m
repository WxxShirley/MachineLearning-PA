function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

x_ = X';
temp = theta' * x_ ;
h_theta = sigmoid(temp);
mul1 = log(h_theta);
y_T = y';
mul1 = (-y_T).*mul1;
mul2 = log(ones(size(h_theta))-h_theta);
mul2 = -((ones(size(y_T))-y_T)).*mul2;
J = sum(mul1+mul2)/m;

for i=1:3,
          grad(i,1) = (sum((h_theta - y_T).* x_(i,:)))/m;
end;
          
          







% =============================================================

end
