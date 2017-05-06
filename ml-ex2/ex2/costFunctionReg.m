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


z = X * theta;
h = sigmoid(z);

lp = (-y') * log(h);
rp = (1-y)' * log(1-h);

theta_zero = theta;

# indexed from 1 to m and skipping theta zero (index = 1)
theta_zero(1) = 0;

lambda_J_cost = (lambda / (2*m)) * sum(theta_zero .^ 2);
lambda_grad_cost = lambda / m * theta_zero;

% cost: J
J = (1/m) * (lp - rp) + lambda_J_cost;

% gradient: compute as the derivative of the cost function
grad = (1/m) * X' * (h-y) + lambda_grad_cost;


% =============================================================

end
