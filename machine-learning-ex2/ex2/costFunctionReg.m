function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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


x_ = X';
temp = theta' * x_ ;
h_theta = sigmoid(temp);
mul1 = log(h_theta);
y_T = y';
mul1 = (-y_T).*mul1;
mul2 = log(ones(size(h_theta))-h_theta);
mul2 = -((ones(size(y_T))-y_T)).*mul2;
t_ = sum(mul1+mul2)/m;

temp_ = theta' * theta - theta(1)*theta(1);
J = t_ + temp_*lambda/(2*m);

len = length(theta)

grad(1) = (sum((h_theta - y_T).* x_(1,:)))/m ;

for i=2:len,
grad(i) = (sum((h_theta - y_T).* x_(i,:)))/m + lambda * theta(i)/m;
%end;



% =============================================================

end
