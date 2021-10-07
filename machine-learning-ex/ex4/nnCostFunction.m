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

% Setup y for each output element

y = [y zeros(m, num_labels - 1)];
for i = 1:m
    z = zeros(1, num_labels);
    for j = 1:num_labels
        z(j) = y(i, 1) == j;
    end
    y(i, :) = z;
end

% Forward Propagation
Delta_1 = zeros(hidden_layer_size, input_layer_size + 1);
Delta_2 = zeros(num_labels, hidden_layer_size + 1);

costs = zeros(m, num_labels);
for i = 1:m
    a_1 = X(i, :);
    z_2 = Theta1 * [ones(1, 1) a_1]';
    a_2 = sigmoid(z_2);
    z_3 = Theta2 * [ones(1,1) a_2']';
    a_3 = sigmoid(z_3);
    sigmoidGradient(z_2);
    h = a_3';

    for j = 1:num_labels
        costs(i, j) = (-y(i ,j) * log(h(j))) - ((1 - y(i,j)) * log(1 - h(j)));
    end    
    
    % Count Gradient (Back Propagation)
    delta_3 = zeros(1, num_labels);
    for j = 1:num_labels
        delta_3(j) = h(j) - y(i,j);
    end
    
    
    delta_2_unweighted = (Theta2' * delta_3') .* sigmoidGradient([ones(size(delta_3', 2), 1) z_2'])';
    delta_2 = delta_2_unweighted(2:end);
    
    
    Delta_2 = Delta_2 + delta_3' * [ones(size(delta_3', 2), 1) a_2'];
    Delta_1 = Delta_1 + delta_2 * [ones(size(delta_2, 2), 1) a_1];
end
sum(sum(costs))
J_unreg = 1/m * sum(sum(costs, 2));
Theta1_reg = Theta1(:, 2:input_layer_size + 1);
Theta2_reg = Theta2(:, 2:hidden_layer_size + 1);
reg = lambda/(2*m) * (sum(sum(Theta1_reg .^ 2)) + sum(sum(Theta2_reg .^ 2)));

J = J_unreg + reg;

Theta1_grad = Delta_1/m;
Theta2_grad = Delta_2/m;

for i = 1:size(Theta1_grad, 1)
    for j = 2:size(Theta1_grad,2)
        Theta1_grad(i,j) = Theta1_grad(i,j) + lambda/m * Theta1(i,j);
    end
end

for i = 1:size(Theta2_grad, 1)
    for j = 2:size(Theta2_grad,2)
        Theta2_grad(i,j) = Theta2_grad(i,j) + lambda/m * Theta2(i,j);
    end
end


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
%
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



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
