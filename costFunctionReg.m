function [J, grad] = costFunctionReg(theta, X, y, lambda)
%costFunctionReg Compute cost for linear regression
%   J = costFunctionReg(theta, X, y, lambda) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y 

% Initialize some useful values
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));
h = X*theta;
J = sum((h-y).^2)/(2*m)+lambda*sum(theta.^2)/(2*m);
temp = lambda*theta/m; 
temp(1) = 0;   
grad = X'*(h-y)/m+temp;
grad = grad(:);

end
