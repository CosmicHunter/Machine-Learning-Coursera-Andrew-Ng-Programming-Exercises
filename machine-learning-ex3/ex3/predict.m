function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.

  %  layer1 (input)  = 400 nodes + 1bias
 % layer2 (hidden) = 25 nodes + 1bias 
  % layer3 (output) = 10 nodes
  
  % dimension of theta =  sj+1 * (sj + 1)
  % theta1 will have a dimension of 25 x 401
  % theta2 will have a dimension of 10 x 26
 % X = [ones(m,1) X];
  % X is of m x  no of features in each image
  %X has dimension of := no of images  x no of features for each image
  % no of features me ek 1 daala aur a1 bana x0 add kia a1 ko hi X likha hai
 %a2 = sigmoid(X * (Theta1)');
 %a2 = [ones(size(a2,1),1) a2];
 %a3 = sigmoid(a2 * (Theta2)');
 %[predictionforeverytrainingex,class] = max(a3,[],2); 
 % a3 will be no of training ex X no of classes  % so we  need to take row wise maximum
 % the index at which maximum is found is the class number  

a1 = [ones(m,1) X]; % 5000 x 401 == no_of_input_images x no_of_features % Adding 1 in X 
  %No. of rows = no. of input images
  %No. of Column = No. of features in each image
  
  z2 = a1 * Theta1';  % 5000 x 25
  a2 = sigmoid(z2);   % 5000 x 25
 
  a2 =  [ones(size(a2,1),1) a2];  % 5000 x 26
  
  z3 = a2 * Theta2';  % 5000 x 10
  a3 = sigmoid(z3);  % 5000 x 10
  
  [prob, p] = max(a3,[],2); 






% =========================================================================


end
