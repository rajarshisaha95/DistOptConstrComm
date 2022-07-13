% This script generates synthetic data for classification using support
% vector machine

clear all;
close all;
clc;

num_dims = 30;                      % Dimension of the problem
num_data = 100;                     % Total no. of data points

x_act = randn(num_dims,1);          % Nominal separating hyperplane
x_act = x_act/norm(x_act,2);


% Nominal center of class +1 data
z_1 = randn(num_dims,1);
z_1 = z_1/norm(z_1,2);
lim = 0.7;
while (x_act'*z_1) < lim
    z_1 = randn(num_dims,1);
    z_1 = z_1/norm(z_1,2);
end

% Nominal center of class -1 data
z_2 = randn(num_dims,1);
z_2 = z_2/norm(z_2,2);
while (x_act'*z_2) > -lim
    z_2 = randn(num_dims,1);
    z_2 = z_2/norm(z_2,2);
end

% Generate data for class +1
data1 = zeros(num_data/2, num_dims);
for i = 1:1:num_data/2
    data1(i,:) = z_1' + 0.6*randn(1, num_dims);
end

% Generate data for class -1
data2 = zeros(num_data/2, num_dims);
for i = 1:1:num_data/2
    data2(i,:) = z_2' + 0.6*randn(1, num_dims);
end

data = [data1; data2];
y_class = sign(data*x_act);
    
all_data = [data, y_class];

% Shuffle the dataset
all_data_shuffled = all_data(randperm(size(all_data,1)),:);

%Save the dataset
save('SVM_dataset.mat');