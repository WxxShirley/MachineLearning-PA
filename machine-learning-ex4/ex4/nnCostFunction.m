function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%Compute h(Feedforward)
X = [ones(m,1) X];
z2 = X * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m,1) a2];
z3 = a2 * Theta2';
h = sigmoid(z3);
a3 = h;

%Compute CostFunction
y_ = zeros(m,num_labels);
for i=1:m
y_(i,y(i))=1;
end

%J = -(1/m) * sum(sum(y_ .* log(h) + (1-y_) .* log(1-h)));
temp_matrix = zeros(m,1);
for i=1:m,
temp_matrix(i) = -log(h(i,:)*y_'(:,i)) - log(1-h(i,:))*(1-y_'(:,i));
end

J = sum(temp_matrix)/m;

%Compute Regularized cost function
Theta1_temp = Theta1;
for i=1:size(Theta1_temp,1),
Theta1_temp(i)=0;
end
temp1 = sum(Theta1_temp .^ 2);
temp1 = sum(temp1);

Theta2_temp = Theta2;
for i=1:size(Theta2_temp,1),
Theta2_temp(i)=0;
end
temp2 = sum(Theta2_temp .^ 2);
temp2 = sum(temp2);

J += lambda * (temp1+temp2) /2/m;

for ex=1:m
a1=X(ex,:);
a1=a1';
z2=Theta1*a1;
a2=[1;sigmoid(z2)];
z3=Theta2*a2;
a3=sigmoid(z3);
y=y_(ex,:);
delta3=a3-y';
delta2 = Theta2(:,2:end)' * delta3 .* sigmoidGradient(z2);  % delta2 is a 25x1 column vector
Theta1_grad = Theta1_grad + delta2 * a1';
Theta2_grad = Theta2_grad + delta3 * a2';
end

Theta1_grad=Theta1_grad ./ m;
Theta2_grad=Theta2_grad ./ m;

Theta1(:,1) = 0;
Theta2(:,1) = 0;
Theta1_grad = Theta1_grad + lambda / m * Theta1;
Theta2_grad = Theta2_grad + lambda / m * Theta2;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
