function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

c_list = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
s_list = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
count = length(c_list);
errors = zeros(count, count);

for i = 1:count
	for j = 1: count
		model= svmTrain(X, y, c_list(i), @(x1, x2) gaussianKernel(x1, x2, s_list(j))); 
		predictions = svmPredict(model, Xval);
		errors(i,j) = mean(double(predictions ~= yval));
	end;
end;
errors
error_list = errors(:);
[min_err, idx] = min(error_list);

s_idx = ceil(idx/count);
c_idx = mod(idx - 1,  count) + 1;
C = c_list(c_idx);
sigma = s_list(s_idx);

% =========================================================================

end
