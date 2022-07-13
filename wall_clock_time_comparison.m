% This script compares the wallclock time for computing near-democratic
% representation of a vector versus the democratic representation

clear all;
close all;
clc;

num_realizations = 10;                           % No. of realizations for each dimension

n_array = [10, 30, 50, 75, 100, 150, 200, 400, 500, 700, 800, 1000];        % Dimensions
N_array = [16, 32, 64, 128, 128, 256, 256, 512, 512, 1024, 1024, 1024];     % Nearest power of 2
num_dims = 12;                                   % Length of dim_array

elapsed_time_D_Hadamard_array = zeros(num_realizations, num_dims);
elapsed_time_ND_Hadamard_array = zeros(num_realizations, num_dims);
elapsed_time_D_orthonormal_array = zeros(num_realizations, num_dims);
elapsed_time_ND_orthonormal_array = zeros(num_realizations, num_dims);

% Iterate over dimensions
for i = 1:1:num_dims
    
    % Do different realizations
    for realiz_ind = 1:1:num_realizations
        
        % Dimensions
        n = n_array(i);
        N = N_array(i);
        
        % To track progress
        fprintf('\nRealization: %d, n = %d', realiz_ind, n);
        
        % Generate a random vector
        y = randn(n,1).^3;
        
        % Generating a randomized Hadamard frame
        D = diag(2*(randi([0,1], N, 1) - 0.5));   %Post-multiplication random diagonal matrix
        H = (1/sqrt(N))*hadamard(N);
        Id = eye(N);
        perm_rows = randperm(N);
        P = Id(perm_rows(1:n),:);                 %Matrix for randomly selecting rows
        S = P*D*H;
        
        % Wall clock time for democratic representation using randomized
        % Hadamard frame
        timer_val_D_Hadamard = tic;
        cvx_begin quiet
            variable x(N)
            minimize norm(x, Inf)
            subject to
                y == S*x;
        cvx_end
        elapsed_time_D_Hadamard_array(realiz_ind, i) = toc(timer_val_D_Hadamard);
        
        
        % Wall clock time for near-democratic representation using
        % randomized Hadamard frame
        timer_val_ND_Hadamard = tic;
        x = S'*y;
        elapsed_time_ND_Hadamard_array(realiz_ind, i) = toc(timer_val_ND_Hadamard);
        
        
        % Generating a random orthonormal frame
        rand_matrix = randn(N);
        [U, Sigma, V] = svd(rand_matrix);
        rand_orth = U*V';
        random_perm = randperm(N);
        S = rand_orth(random_perm(1:n),:);          %Columns constitute the tight frame
        
        
        % Wall clock time for democratic representation using random
        % orthonormal frame
        timer_val_D_orthonormal = tic;
        cvx_begin quiet
            variable x(N)
            minimize norm(x, Inf)
            subject to
                y == S*x;
        cvx_end
        elapsed_time_D_orthonormal_array(realiz_ind, i) = toc(timer_val_D_orthonormal);
        
        
        % Wall clock time for near democratic representation using random
        % orthonormal frame
        timer_val_ND_orthonormal = tic;
        x = S'*y;
        elapsed_time_ND_orthonormal_array(realiz_ind, i) = toc(timer_val_ND_orthonormal);
        
    end
end


filename = 'wallclock_time_comparison.mat';
save(filename)

%%
% Plot results

elapsed_time_D_Hadamard = mean(elapsed_time_D_Hadamard_array, 1);
elapsed_time_ND_Hadamard = mean(elapsed_time_ND_Hadamard_array, 1);
elapsed_time_D_orthonormal = mean(elapsed_time_D_orthonormal_array, 1);
elapsed_time_ND_orthonormal = mean(elapsed_time_ND_orthonormal_array, 1);

% Computing error bars
err_pos_D_Hadamard = max(elapsed_time_D_Hadamard_array,[], 1) - elapsed_time_D_Hadamard;
err_neg_D_Hadamard = elapsed_time_D_Hadamard - min(elapsed_time_D_Hadamard_array,[], 1);
err_pos_ND_Hadamard = max(elapsed_time_ND_Hadamard_array,[], 1) - elapsed_time_ND_Hadamard;
err_neg_ND_Hadamard = elapsed_time_ND_Hadamard - min(elapsed_time_ND_Hadamard_array,[], 1);
err_pos_D_orthonormal = max(elapsed_time_D_orthonormal_array,[], 1) - elapsed_time_D_orthonormal;
err_neg_D_orthonormal = elapsed_time_D_orthonormal - min(elapsed_time_D_orthonormal_array,[], 1);
err_pos_ND_orthonormal = max(elapsed_time_ND_orthonormal_array,[], 1) - elapsed_time_ND_orthonormal;
err_neg_ND_orthonormal = elapsed_time_ND_orthonormal - min(elapsed_time_ND_orthonormal_array,[], 1);

figure;
errorbar(n_array, elapsed_time_D_Hadamard, err_neg_D_Hadamard, err_pos_D_Hadamard, '--');
hold on;
errorbar(n_array, elapsed_time_ND_Hadamard, err_neg_ND_Hadamard, err_pos_ND_Hadamard, '-');
errorbar(n_array, elapsed_time_D_orthonormal, err_neg_D_orthonormal, err_pos_D_orthonormal, '--');
errorbar(n_array, elapsed_time_ND_orthonormal, err_neg_ND_orthonormal, err_pos_ND_orthonormal, '-');
legend('Democratic (Hadamard)', 'Near-Democratic (Hadamard)', 'Democratic (Orthonormal)', 'Near-Democratic (Orthonormal)');
