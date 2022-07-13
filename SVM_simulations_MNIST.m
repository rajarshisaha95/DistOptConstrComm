% This script compares performance of different source coding schemes when
% a support vector machine is trained using quantized gradients from a
% noisy subgradient oracle for digit 0 vs. 1 classification from the MNIST
% dataset.

clear all;
close all;
clc;

% Load the MNIST dataset
load('mnist_all.mat');
train0 = im2double(train0);
train1 = im2double(train1);
test0 = im2double(test0);
test1 = im2double(test1);

num_0s_train = size(train0,1);
num_1s_train = size(train1,1);
num_0s_test = size(test0,1);
num_1s_test = size(test1,1);

label_0s_train = -1*ones(num_0s_train,1);
label_1s_train = 1*ones(num_1s_train,1);
label_0s_test = -1*ones(num_0s_test,1);
label_1s_test = 1*ones(num_1s_test,1);

train_samples = [train0;train1];
test_samples = [test0;test1];
train_labels = [label_0s_train;label_1s_train];
test_labels = [label_0s_test;label_1s_test];

data_train = [train_samples, train_labels];
data_test = [test_samples, test_labels];

% Shuffle the dataset
data = data_train(randperm(size(data_train,1)),:);

num_dims = size(train0, 2);
num_data = size(data, 1);

% Generate CVX output
lambda = 0;

% cvx_begin
% 
%     variable x(num_dims)
%     t = max(1 - data(:,num_dims+1).*(data(:,1:num_dims)*x),0);
%     minimize 1/num_data*sum(t) + lambda/(2*num_data)*quad_over_lin(x,1)
%     
% cvx_end

num_realizations = 1;                   % No. of ensemble realizations for which the results are averaged
max_iters = 500;                       % Maximum number of iterations in each realization

% To ensure common initialization 
x_init = randn(num_dims,1);

%To store average over different ensemble realizations
obj_fnval_avg = zeros(1, max_iters);                       
classification_error_avg = zeros(1, max_iters);            

obj_fnval_avg_QPSGD = zeros(1, max_iters);                 
classification_error_avg_QPSGD = zeros(1, max_iters);      

obj_fnval_avg_NDQPSGD = zeros(1, max_iters);                
classification_error_avg_NDQPSGD = zeros(1, max_iters);

obj_fnval_avg_topK_QPSGD = zeros(1, max_iters);
classification_error_avg_topK_QPSGD = zeros(1, max_iters);

obj_fnval_avg_topK_NDQPSGD = zeros(1, max_iters);
classification_error_avg_topK_NDQPSGD = zeros(1, max_iters);

eta = 1;              % Step size

%%
% Quantizer paramaters
R = 0.1;                                % Data rate per dimension
total_num_bits = floor(num_dims*R);

% Allocation of total_num_bits to each dimension (for vanilla scalar quantizer)
num_bits_per_dim = floor(total_num_bits/num_dims);             
num_bits = num_bits_per_dim*ones(1,num_dims);                      % Initial allocation
remaining_bits = total_num_bits - num_dims*num_bits_per_dim;
rand_permutation = randperm(num_dims);
for j = 1:1:remaining_bits
    num_bits(rand_permutation(j)) = num_bits(rand_permutation(j)) + 1;
end

num_points_per_dim = 2.^num_bits;                   % Number of points per dimension
resolution_per_dim = 2./num_points_per_dim;         % Length of interval in each dimension


%% 
% Quantizer paramaters (when top-K sparsification is also used)
R = 0.1;                                % Data rate per dimension
total_num_bits_topk = floor(num_dims*R);

% Allocation of total number of bits to each dimension of the sparsified gradient in top-K
frac = 0.1;
K = floor(frac*num_dims);

num_bits_per_dim_topk = floor(total_num_bits_topk/K);
num_bits_topk = num_bits_per_dim_topk*ones(1,K);                    % Initial allocation
remaining_bits = total_num_bits_topk - K*num_bits_per_dim_topk;
rand_permutation = randperm(K);
for j = 1:1:remaining_bits
    num_bits_topk(rand_permutation(j)) = num_bits_topk(rand_permutation(j)) + 1;
end

num_points_per_dim_topk = 2.^num_bits_topk;                   % Number of points per dimension
resolution_per_dim_topk = 2./num_points_per_dim_topk;         % Length of interval in each dimension

%%
% Unquantized subgradient descent
for realiz_ind = 1:1:num_realizations
    
    % Initialization
    x = x_init;
    x_av = x_init;                                  % Iteration averaged primal iterate
    
    % Global objective (and classification error) evaluated at the iterates of each node
    obj_fnval = zeros(1, max_iters);
    classification_error = zeros(1, max_iters);
    
    % To store the minimum objective value so far
    min_obj_val = +Inf;
    
    for k = 1:1:max_iters
       
        % To track progress
        k
        
        % Store normed hinge loss for plotting later
        obj_fnval_curr = mean(max(1 - data(:,num_dims+1).*(data(:,1:num_dims)*x_av),0)) + lambda/(2*num_data)*norm(x_av,2)^2;
        obj_fnval(1,k) = min(min_obj_val, obj_fnval_curr);
        min_obj_val = obj_fnval(1,k);
        
        % Store classfication error
        pred = sign(data_test(:,1:num_dims)*x_av);
        classification_error(1,k) = mean(pred.*data_test(:,num_dims+1) < 0);
        
        % Evaluate stochastic subgradient
        % Randomly sampling datapoint 
        frac_data = 0.7;                % Fraction of local dataset used for computing (sub)gradient
        num_indices = floor(frac_data*num_data);
        idx = randperm(num_data, num_indices);
        sampled_data = data(idx,:);
        
        %Computing stochastic (sub)gradient
        is_active = sampled_data(:,num_dims+1).*(sampled_data(:,1:num_dims)*x_av) < 1;

        temp = zeros(num_indices, num_dims);
        for j = 1:1:num_indices
            temp(j,:) = is_active(j)*(sampled_data(j,num_dims+1)*sampled_data(j,1:num_dims));
        end
        grad = -mean(temp, 1)' + x_av/num_data;
        
        % Taking a subgradient step
        x = x - eta*grad;
        
        % Update running average
        x_av = (1/k)*((k-1)*x_av + x);
      
    end
    
    obj_fnval_avg = obj_fnval_avg + obj_fnval;
    classification_error_avg = classification_error_avg + classification_error;
    
end

obj_fnval_avg = obj_fnval_avg/num_realizations;
classification_error_avg = classification_error_avg/num_realizations;

%%
% Vanilla Quantized Projected Subgradient Descent

for realiz_ind = 1:1:num_realizations
    
    % Initialization
    x = x_init;
    x_av = x_init;                                  % Iteration averaged primal iterate
    
    % Global objective (and classification error) evaluated at the iterates of each node
    obj_fnval = zeros(1, max_iters);
    classification_error = zeros(1, max_iters);
    
    % To store the minimum objective value so far
    min_obj_val = +Inf;
    
    for k = 1:1:max_iters
       
        % To track progress
        k
        
        % Store normed hinge loss for plotting later
        obj_fnval_curr = mean(max(1 - data(:,num_dims+1).*(data(:,1:num_dims)*x_av),0)) + lambda/(2*num_data)*norm(x_av,2)^2;
        obj_fnval(1,k) = min(min_obj_val, obj_fnval_curr);
        min_obj_val = obj_fnval(1,k);
        
        % Store classfication error
        pred = sign(data_test(:,1:num_dims)*x_av);
        classification_error(1,k) = mean(pred.*data_test(:,num_dims+1) < 0);
        
        % Evaluate stochastic subgradient
        % Randomly sampling datapoint 
        frac_data = 0.7;                % Fraction of local dataset used for computing (sub)gradient
        num_indices = floor(frac_data*num_data);
        idx = randperm(num_data, num_indices);
        sampled_data = data(idx,:);
        
        % Computing stochastic (sub)gradient
        is_active = sampled_data(:,num_dims+1).*(sampled_data(:,1:num_dims)*x_av) < 1;

        temp = zeros(num_indices, num_dims);
        for j = 1:1:num_indices
            temp(j,:) = is_active(j)*(sampled_data(j,num_dims+1)*sampled_data(j,1:num_dims));
        end
        grad = -mean(temp, 1)' + x_av/num_data;
        
        % Compress the gradient
        q_grad = zeros(num_dims,1);            % Initializing the quantized coefficients
        dyn_range = norm(grad, Inf);
        grad_scaled = 1/dyn_range*grad;

        % Quantize each dimension of the scaled input with scalar quantizer of unit dynamic range
        for j = 1:1:num_dims
            
            if (num_bits(j) == 0)
                
                q_grad(j) = 0;
                
            else
                
                DELTA = 1/(2^num_bits(j)-1);
            
                ind = floor(abs(grad_scaled(j))/DELTA);
                lower_point = ind*DELTA;
                upper_point = (ind+1)*DELTA;

                p = (abs(grad_scaled(j)) - lower_point)/DELTA;

                if(rand > p)
                    q_grad(j) = dyn_range*sign(grad_scaled(j))*lower_point;
                else
                    q_grad(j) = dyn_range*sign(grad_scaled(j))*upper_point;
                end 
                
            end 
            
        end

        % Taking a subgradient step
        x = x - eta*q_grad;
        
        % Update running average
        x_av = (1/k)*((k-1)*x_av + x);
      
    end
    
    obj_fnval_avg_QPSGD = obj_fnval_avg_QPSGD + obj_fnval;
    classification_error_avg_QPSGD = classification_error_avg_QPSGD + classification_error;
    
end

obj_fnval_avg_QPSGD = obj_fnval_avg_QPSGD/num_realizations;
classification_error_avg_QPSGD = classification_error_avg_QPSGD/num_realizations;


%%
% Projected subgradient descent with near-democratic + compression
for realiz_ind = 1:1:num_realizations
    
    % Initialization
    x = x_init;
    x_av = x_init;                                  % Iteration averaged primal iterate
    
    % Global objective (and classification error) evaluated at the iterates of each node
    obj_fnval = zeros(1, max_iters);
    classification_error = zeros(1, max_iters);
    
    % To store the minimum objective value so far
    min_obj_val = +Inf;
    
    for k = 1:1:max_iters
       
        % To track progress
        k
        
        % Store normed hinge loss for plotting later
        obj_fnval_curr = mean(max(1 - data(:,num_dims+1).*(data(:,1:num_dims)*x_av),0)) + lambda/(2*num_data)*norm(x_av,2)^2;
        obj_fnval(1,k) = min(min_obj_val, obj_fnval_curr);
        min_obj_val = obj_fnval(1,k);
        
        % Store classfication error
        pred = sign(data_test(:,1:num_dims)*x_av);
        classification_error(1,k) = mean(pred.*data_test(:,num_dims+1) < 0);
        
        % Evaluate stochastic subgradient
        % Randomly sampling datapoint 
        frac_data = 0.7;                % Fraction of local dataset used for computing (sub)gradient
        num_indices = floor(frac_data*num_data);
        idx = randperm(num_data, num_indices);
        sampled_data = data(idx,:);
        
        %Computing stochastic (sub)gradient
        is_active = sampled_data(:,num_dims+1).*(sampled_data(:,1:num_dims)*x_av) < 1;

        temp = zeros(num_indices, num_dims);
        for j = 1:1:num_indices
            temp(j,:) = is_active(j)*(sampled_data(j,num_dims+1)*sampled_data(j,1:num_dims));
        end
        grad = -mean(temp, 1)' + x_av/num_data;
        
        % Generate a random orthonormal frame democratic representation
        rand_matrix = randn(num_dims);
        [U, Sigma, V] = svd(rand_matrix);
        rand_orth = U*V';
        random_perm = randperm(num_dims);
        S = rand_orth(random_perm(1:num_dims),:);          % Columns constitute the tight frame
        
        % Find near democratic representation
        grad_coeff = S'*grad;
        
        % Compress the gradient
        q_coeff = zeros(num_dims,1);            % Initializing the quantized coefficients
        dyn_range = norm(grad_coeff, Inf);
        grad_coeff_scaled = 1/dyn_range*grad_coeff;

        %Quantize each dimension of the scaled input with scalar quantizer of unit dynamic range
        for j = 1:1:num_dims
            
             if (num_bits(j) == 0)
                
                q_coeff(j) = 0;
                
             else
                 
                DELTA = 1/(2^num_bits(j)-1);
            
                ind = floor(abs(grad_coeff_scaled(j))/DELTA);
                lower_point = ind*DELTA;
                upper_point = (ind+1)*DELTA;

                p = (abs(grad_coeff_scaled(j)) - lower_point)/DELTA;

                if(rand > p)
                    q_coeff(j) = dyn_range*sign(grad_coeff_scaled(j))*lower_point;
                else
                    q_coeff(j) = dyn_range*sign(grad_coeff_scaled(j))*upper_point;
                end
                 
             end
            
        end

        q_ndo_grad = S*q_coeff;                          % Inverse transform from quantized coefficients

        % Taking a subgradient step
        x = x - eta*q_ndo_grad;
        
        % Update running average
        x_av = (1/k)*((k-1)*x_av + x);
      
    end
    
    obj_fnval_avg_NDQPSGD = obj_fnval_avg_NDQPSGD + obj_fnval;
    classification_error_avg_NDQPSGD = classification_error_avg_NDQPSGD + classification_error;
    
end

obj_fnval_avg_NDQPSGD = obj_fnval_avg_NDQPSGD/num_realizations;
classification_error_avg_NDQPSGD = classification_error_avg_NDQPSGD/num_realizations;

%%
% Top-K sparsification + Vanilla Quantized Projected Subgradient Descent

for realiz_ind = 1:1:num_realizations
    
    % Initialization
    x = x_init;
    x_av = x_init;                                  % Iteration averaged primal iterate
    
    % Global objective (and classification error) evaluated at the iterates of each node
    obj_fnval = zeros(1, max_iters);
    classification_error = zeros(1, max_iters);
    
    % To store the minimum objective value so far
    min_obj_val = +Inf;
    
    for k = 1:1:max_iters
       
        % To track progress
        k
        
        % Store normed hinge loss for plotting later
        obj_fnval_curr = mean(max(1 - data(:,num_dims+1).*(data(:,1:num_dims)*x_av),0)) + lambda/(2*num_data)*norm(x_av,2)^2;
        obj_fnval(1,k) = min(min_obj_val, obj_fnval_curr);
        min_obj_val = obj_fnval(1,k);
        
        % Store classfication error
        pred = sign(data_test(:,1:num_dims)*x_av);
        classification_error(1,k) = mean(pred.*data_test(:,num_dims+1) < 0);
        
        % Evaluate stochastic subgradient
        % Randomly sampling datapoint 
        frac_data = 0.7;                % Fraction of local dataset used for computing (sub)gradient
        num_indices = floor(frac_data*num_data);
        idx = randperm(num_data, num_indices);
        sampled_data = data(idx,:);
        
        % Computing stochastic (sub)gradient
        is_active = sampled_data(:,num_dims+1).*(sampled_data(:,1:num_dims)*x_av) < 1;

        temp = zeros(num_indices, num_dims);
        for j = 1:1:num_indices
            temp(j,:) = is_active(j)*(sampled_data(j,num_dims+1)*sampled_data(j,1:num_dims));
        end
        grad = -mean(temp, 1)' + x_av/num_data;
      
        % Sparsify the gradient
        % Top-k sparsification step
        [grad_sorted, indices] = sort(abs(grad), 'descend');              % Sort the elements in ascending order of magnitude
        grad_topk_trunc = grad(indices(1:K));  
        
        % Compress the gradient
        q_grad_topk_trunc = zeros(K,1);            % Initializing the quantized coefficients
        dyn_range = norm(grad_topk_trunc, Inf);
        grad_topk_trunc_scaled = 1/dyn_range*grad_topk_trunc;

        % Quantize each dimension of the scaled input with scalar quantizer of unit dynamic range
        for j = 1:1:K

            if (num_bits_topk(j) == 0)
                
                q_grad_topk_trunc = 0;
                
            else
                
                DELTA = 1/(2^num_bits_topk(j)-1);
            
                ind = floor(abs(grad_topk_trunc_scaled(j))/DELTA);
                lower_point = ind*DELTA;
                upper_point = (ind+1)*DELTA;

                p = (abs(grad_topk_trunc_scaled(j)) - lower_point)/DELTA;

                if(rand > p)
                    q_grad_topk_trunc(j) = dyn_range*sign(grad_topk_trunc_scaled(j))*lower_point;
                else
                    q_grad_topk_trunc(j) = dyn_range*sign(grad_topk_trunc_scaled(j))*upper_point;
                end 
            end      
        end

        % Fill in the zeros
        q_grad_topk = zeros(num_dims,1);
        for j = 1:1:K
            q_grad_topk(indices(j)) = q_grad_topk_trunc(j);
        end 

        % Taking a subgradient step
        x = x - eta*q_grad_topk;
        
        % Update running average
        x_av = (1/k)*((k-1)*x_av + x);
      
    end
    
    obj_fnval_avg_topK_QPSGD = obj_fnval_avg_topK_QPSGD + obj_fnval;
    classification_error_avg_topK_QPSGD = classification_error_avg_topK_QPSGD + classification_error;
    
end

obj_fnval_avg_topK_QPSGD = obj_fnval_avg_topK_QPSGD/num_realizations;
classification_error_avg_topK_QPSGD = classification_error_avg_topK_QPSGD/num_realizations;


%%
% Top-K sparsification + Near-democratic + Quantized Projected Subgradient Descent

for realiz_ind = 1:1:num_realizations
    
    % Initialization
    x = x_init;
    x_av = x_init;                                  % Iteration averaged primal iterate
    
    % Global objective (and classification error) evaluated at the iterates of each node
    obj_fnval = zeros(1, max_iters);
    classification_error = zeros(1, max_iters);
    
    % To store the minimum objective value so far
    min_obj_val = +Inf;
    
    for k = 1:1:max_iters
       
        % To track progress
        k
        
        % Store normed hinge loss for plotting later
        obj_fnval_curr = mean(max(1 - data(:,num_dims+1).*(data(:,1:num_dims)*x_av),0)) + lambda/(2*num_data)*norm(x_av,2)^2;
        obj_fnval(1,k) = min(min_obj_val, obj_fnval_curr);
        min_obj_val = obj_fnval(1,k);
        
        % Store classfication error
        pred = sign(data_test(:,1:num_dims)*x_av);
        classification_error(1,k) = mean(pred.*data_test(:,num_dims+1) < 0);
        
        % Evaluate stochastic subgradient
        % Randomly sampling datapoint 
        frac_data = 0.7;                % Fraction of local dataset used for computing (sub)gradient
        num_indices = floor(frac_data*num_data);
        idx = randperm(num_data, num_indices);
        sampled_data = data(idx,:);
        
        % Computing stochastic (sub)gradient
        is_active = sampled_data(:,num_dims+1).*(sampled_data(:,1:num_dims)*x_av) < 1;

        temp = zeros(num_indices, num_dims);
        for j = 1:1:num_indices
            temp(j,:) = is_active(j)*(sampled_data(j,num_dims+1)*sampled_data(j,1:num_dims));
        end
        grad = -mean(temp, 1)' + x_av/num_data;
        
      
        % Sparsify the gradient
        % Top-k sparsification step
        [grad_sorted, indices] = sort(abs(grad), 'descend');              % Sort the elements in ascending order of magnitude
        grad_topk_trunc = grad(indices(1:K));
        
        % Generate a random orthonormal frame democratic representation
        rand_matrix = randn(K);
        [U, Sigma, V] = svd(rand_matrix);
        rand_orth = U*V';
        random_perm = randperm(K);
        S = rand_orth(random_perm(1:K),:);          % Columns constitute the tight frame
        
        % Find near democratic representation
        grad_coeff = S'*grad_topk_trunc;
        
        % Compress the gradient
        q_coeff = zeros(K,1);            % Initializing the quantized coefficients
        dyn_range = norm(grad_coeff, Inf);
        grad_coeff_scaled = 1/dyn_range*grad_coeff;

        % Quantize each dimension of the scaled input with scalar quantizer of unit dynamic range
        for j = 1:1:K

            if (num_bits_topk(j) == 0)
                
                q_coeff = 0;
                
            else
                
                DELTA = 1/(2^num_bits_topk(j)-1);
            
                ind = floor(abs(grad_coeff_scaled(j))/DELTA);
                lower_point = ind*DELTA;
                upper_point = (ind+1)*DELTA;

                p = (abs(grad_coeff_scaled(j)) - lower_point)/DELTA;

                if(rand > p)
                    q_coeff(j) = dyn_range*sign(grad_coeff_scaled(j))*lower_point;
                else
                    q_coeff(j) = dyn_range*sign(grad_coeff_scaled(j))*upper_point;
                end 
            end      
        end

        % Compute the inverse transform
        q_grad_topk_trunc = S*q_coeff;
        
        % Fill in the zeros
        q_grad_topk = zeros(num_dims,1);
        for j = 1:1:K
            q_grad_topk(indices(j)) = q_grad_topk_trunc(j);
        end 
        
        % Taking a subgradient step
        x = x - eta*q_grad_topk;
        
        % Update running average
        x_av = (1/k)*((k-1)*x_av + x);
      
    end
    
    obj_fnval_avg_topK_NDQPSGD = obj_fnval_avg_topK_NDQPSGD + obj_fnval;
    classification_error_avg_topK_NDQPSGD = classification_error_avg_topK_NDQPSGD + classification_error;
    
end

obj_fnval_avg_topK_NDQPSGD = obj_fnval_avg_topK_NDQPSGD/num_realizations;
classification_error_avg_topK_NDQPSGD = classification_error_avg_topK_NDQPSGD/num_realizations;


%%

% Plotting results
figure;
x_axis = 1:1:max_iters;
semilogy(x_axis, obj_fnval_avg);
hold on;
semilogy(x_axis, obj_fnval_avg_QPSGD);
semilogy(x_axis, obj_fnval_avg_NDQPSGD);
semilogy(x_axis, obj_fnval_avg_topK_QPSGD);
semilogy(x_axis, obj_fnval_avg_topK_NDQPSGD);
legend('Unquantized', 'Vanilla Quantized', 'NDO-compressed', 'Top-K Vanilla Quant.', 'Top-K + NDO Quant');

figure;
x_axis = 1:1:max_iters;
semilogy(x_axis, classification_error_avg);
hold on;
semilogy(x_axis, classification_error_avg_QPSGD);
semilogy(x_axis, classification_error_avg_NDQPSGD);
semilogy(x_axis, classification_error_avg_topK_QPSGD);
semilogy(x_axis, classification_error_avg_topK_NDQPSGD);
legend('Unquantized', 'Vanilla Quantized', 'NDO-compressed', 'Top-K Vanilla Quant.', 'Top-K + NDO Quant');

