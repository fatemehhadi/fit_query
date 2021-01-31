function [theta, J_history] = gradientDescent_intercept(X, y, theta, alpha, num_iters, lambda)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    h = X*theta;
    J = sum((h-y).^2)/(2*m)+lambda*sum(theta(2:end).^2)/(2*m);
    temp = lambda*theta/m;  
    temp(1) = 0; 
    grad = X'*(h-y)/m+temp;
    theta = theta-alpha*grad;  
    J_history(iter,1) = J;

end

end
