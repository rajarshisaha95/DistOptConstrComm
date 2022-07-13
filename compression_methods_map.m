% This code plots the normalized error of different compression / quantization 
% schemes versus number of bits used for quantization.

clear all;
close all;
clc;

warning('off');

n = 1000;                   % Dimension of the vector being quantized
num_realizations = 50;      % Number of vectors quantized for a fixed data-rate 

min_R = 1.1;                % Minimum number of bits used for quantizing each coordinate
R_step = 0.2;               % Data-rate increment size
max_R = 8;                  % Maximum number of bits used for quantizing each coordinate

% Range of data rates
data_rate_variation = min_R:R_step:max_R;

%%
% Standard dithering

alpha_array_StdDith = zeros(1,length(data_rate_variation));             % To store normalized variance for each data-rate
effective_num_bits_per_dim = zeros(1,length(data_rate_variation));      % Number of bits (including additional bits for sending indices etc.)

R_ind = 0;

for R = min_R:R_step:max_R
    
    R_ind = R_ind + 1;                  % Counter to increment data rate array indexing
    
    alpha_StdDith = 0;                  % Cumulative normalized error (to average out later)
    total_num_bits = floor(n*R);        % Total bit-budget
    
    for realiz_ind = 1:1:num_realizations
        
        % Generate a heavy-tailed random vector
        x = randn(n,1).^3;
        
        % Allocating integral number of bits to each dimension
        num_bits_per_dim = floor(total_num_bits/n);             
        num_bits = num_bits_per_dim*ones(1,n);                      % Initial allocation
        remaining_bits = total_num_bits - n*num_bits_per_dim;
        rand_permutation = randperm(n);
        for j = 1:1:remaining_bits
            num_bits(rand_permutation(j)) = num_bits(rand_permutation(j)) + 1;
        end
        
        % Standard dither quantization
        s = norm(x,Inf);                    
        x_scaled = (1/s)*x;                 % Normalize by the dynamic range
        x_hat = zeros(n,1);                 % Initialize quantized vector
        
        % Do dithering and scalar quantize each coordinate 
        for j = 1:1:n
    
            DELTA = 1/(2^num_bits(j)-1);

            ind = floor(abs(x_scaled(j))/DELTA);
            lower_point = ind*DELTA;
            upper_point = (ind+1)*DELTA;

            p = (abs(x_scaled(j)) - lower_point)/DELTA;

            if(rand > p)
                x_hat(j) = s*sign(x_scaled(j))*lower_point;
            else
                x_hat(j) = s*sign(x_scaled(j))*upper_point;
            end  
            
        end

        % Compute normalized variance for this realization and add it to the
        % cumulative sum to compute empirical average later
        alpha_StdDith = alpha_StdDith + norm(x_hat - x,2)^2/norm(x,2)^2;
        
    end
    
    % Empirical normalized error
    alpha_array_StdDith(R_ind) = alpha_StdDith/num_realizations;
    
    % Effective number of bits per dimension
    effective_num_bits_per_dim(R_ind) = R;
    
    % To track progress
    fprintf('Standard dithering. Data rate = %f\n', R);
    
end

%%
% Near-Democratic representation (randomized Hadamard frame) + Standard dithering

N = 2^(ceil(log2(n)));                                                      % Higher dimension 
alpha_array_NDStdDith = zeros(1, length(data_rate_variation));              % To store normalized variance for each data rate
effective_num_bits_per_dim_ND = zeros(1,length(data_rate_variation));       % Number of bits (including additional bits for sending indices etc.)

R_ind = 0;

for R = min_R:R_step:max_R
    
    R_ind = R_ind + 1;                          % Counter to increment data rate array indexing
 
    alpha_NDStdDith = 0;                        % Cumulative normalized error (to average out later)     
    total_num_bits = floor(n*R);                % Total bit budget
    
    for realiz_ind = 1:1:num_realizations
        
        % Generate a heavy-tailed random vector
        x = randn(n,1).^3;
        
        % Allocating integral number of bits to each dimension
        num_bits_per_dim_NDStdDither = floor(total_num_bits/N);
        num_bits_NDStdDither = num_bits_per_dim_NDStdDither*ones(1,N);    % Initial allocation
        remaining_bits = total_num_bits - N*num_bits_per_dim_NDStdDither;
        rand_permutation = randperm(N);
        for j = 1:1:remaining_bits
            num_bits_NDStdDither(rand_permutation(j)) = num_bits_NDStdDither(rand_permutation(j)) + 1;
        end
        
        % Generate the randomized hadamard frame
        D = diag(2*(randi([0,1], N, 1) - 0.5));         % Random diagonal matrix
        H = (1/sqrt(N))*hadamard(N);
        Id = eye(N);
        perm_rows = randperm(N);
        P = Id(perm_rows(1:n),:);                       % Matrix for randomly selecting rows
        S = P*D*H;
        
        x_coeff = S'*x;                                 % Near-democratic representation
        
        % Standard dither quantization
        s_coeff = norm(x_coeff,Inf);                    % Dynamic range of the coefficients
        x_coeff_scaled = (1/s_coeff)*x_coeff;           % Normalize coefficients to unit dynamic range
        
        x_coeff_hat_NDStdDither = zeros(N,1);           % Initialize quantized coefficient vector
        
        for j = 1:1:N
    
            DELTA = 1/(2^num_bits_NDStdDither(j)-1);

            ind = floor(abs(x_coeff_scaled(j))/DELTA);
            lower_point = ind*DELTA;
            upper_point = (ind+1)*DELTA;

            p = (abs(x_coeff_scaled(j)) - lower_point)/DELTA;

            if(rand > p)
                x_coeff_hat_NDStdDither(j) = s_coeff*sign(x_coeff_scaled(j))*lower_point;
            else
                x_coeff_hat_NDStdDither(j) = s_coeff*sign(x_coeff_scaled(j))*upper_point;
            end

        end
        
        x_hat_NDStdDither = S*x_coeff_hat_NDStdDither;              % Inverse transform
        
        % Compute normalized variance for this realization and add it to the
        % cumulative sum to compute empirical average later
        alpha_NDStdDith = alpha_NDStdDith + norm(x_hat_NDStdDither - x,2)^2/norm(x,2)^2;
        
    end
    
    % Empirical normalized error
    alpha_array_NDStdDith(R_ind) = alpha_NDStdDith/num_realizations;
    
    % Effective number of bits per dimension
    effective_num_bits_per_dim_ND(R_ind) = R;
    
    % To track progress
    fprintf('Near-democratic (random Hadamard frame) + Standard dithering. Data rate = %f\n', R);
     
end

%%
% Near-Democratic representation (random orthonormal frame) + Standard dithering

lambda = 1;                                                                  % Aspect-ratio (redundancy) of the frame
N = ceil(lambda*n);                                                          % Higher dimension
alpha_array_NDOStdDith = zeros(1, length(data_rate_variation));              % To store normalized variance for each data rate
effective_num_bits_per_dim_NDO = zeros(1,length(data_rate_variation));       % Number of bits (including additional bits for sending indices etc.)

R_ind = 0;

for R = min_R:R_step:max_R
    
    R_ind = R_ind + 1;                          % Counter to increment data rate array indexing
 
    alpha_NDOStdDith = 0;                       % Cumulative normalized error (to average out later)     
    total_num_bits = floor(n*R);                % Total bit budget
    
    for realiz_ind = 1:1:num_realizations
        
        % Generate a heavy-tailed random vector
        x = randn(n,1).^3;
        
        % Allocating integral number of bits to each dimension
        num_bits_per_dim_NDOStdDither = floor(total_num_bits/N);
        num_bits_NDOStdDither = num_bits_per_dim_NDOStdDither*ones(1,N);     % Initial allocation
        remaining_bits = total_num_bits - N*num_bits_per_dim_NDOStdDither;
        rand_permutation = randperm(N);
        for j = 1:1:remaining_bits
            num_bits_NDOStdDither(rand_permutation(j)) = num_bits_NDOStdDither(rand_permutation(j)) + 1;
        end
        
        % Generate the random orthonormal frame
        rand_matrix = randn(N);
        [U, Sigma, V] = svd(rand_matrix);
        rand_orth = U*V';
        random_perm = randperm(N);
        S = rand_orth(random_perm(1:n),:);              % Columns constitute the tight frame
        
        x_coeff = S'*x;                                 % Near-democratic representation
        
        %Standard dither quantization
        s_coeff = norm(x_coeff,Inf);                    % Dynamic range of the coefficients
        x_coeff_scaled = (1/s_coeff)*x_coeff;           % Normalize coefficients to unit dynamic range
        
        x_coeff_hat_NDOStdDither = zeros(N,1);          % Initialize quantized coefficient vector
        
        for j = 1:1:N
    
            DELTA = 1/(2^num_bits_NDOStdDither(j)-1);

            ind = floor(abs(x_coeff_scaled(j))/DELTA);
            lower_point = ind*DELTA;
            upper_point = (ind+1)*DELTA;

            p = (abs(x_coeff_scaled(j)) - lower_point)/DELTA;

            if(rand > p)
                x_coeff_hat_NDOStdDither(j) = s_coeff*sign(x_coeff_scaled(j))*lower_point;
            else
                x_coeff_hat_NDOStdDither(j) = s_coeff*sign(x_coeff_scaled(j))*upper_point;
            end

        end
        
        x_hat_NDOStdDither = S*x_coeff_hat_NDOStdDither;              % Inverse transform
        
        % Compute normalized error for this realization and add it to the
        % cumulative sum to compute empirical average later
        alpha_NDOStdDith = alpha_NDOStdDith + norm(x_hat_NDOStdDither - x,2)^2/norm(x,2)^2;
        
    end
    
    % Empirical normalized error
    alpha_array_NDOStdDith(R_ind) = alpha_NDOStdDith/num_realizations;
    
    % Effective number of bits per dimension
    effective_num_bits_per_dim_NDO(R_ind) = R;
    
    % To track progress
    fprintf('Near-democratic (random orthonormal frame) + Standard dithering. Data rate = %f\n', R);
     
end

%%
% Democratic representation (random orthonormal frame - Projected gradient descent) + Standard dithering

lambda = 1.5;
mu = lambda - 1;          % Parameter that controls the redundancy of the frame
constant = 0.3;           % Trades off between UP parameters and probability that the random frame satisfies UP

% Uncertainty principle (UP) parameters
eta = 1 - 0.25*mu;
delta = constant*mu^2/log(1/mu);

N = ceil(lambda*n);           % Dimension of the higher space

alpha_array_DOStdDith = zeros(1, length(data_rate_variation));              % To store normalized variance for each data rate
effective_num_bits_per_dim_DO = zeros(1,length(data_rate_variation));       % Number of bits (including additional bits for sending indices etc.)

R_ind = 0;

for R = min_R:R_step:max_R
    
    R_ind = R_ind + 1;                          % Counter to increment data rate array indexing
 
    alpha_DOStdDith = 0;                        % Cumulative normalized error (to average out later)     
    total_num_bits = floor(n*R);                % Total bit budget
    
    for realiz_ind = 1:1:num_realizations
        
        % Generate a heavy-tailed random vector
        x = randn(n,1).^3;
        
        % Allocating integral number of bits to each dimension
        num_bits_per_dim_DOStdDither = floor(total_num_bits/N);
        num_bits_DOStdDither = num_bits_per_dim_DOStdDither*ones(1,N);      % Initial allocation
        remaining_bits = total_num_bits - N*num_bits_per_dim_DOStdDither;
        rand_permutation = randperm(N);
        for j = 1:1:remaining_bits
            num_bits_DOStdDither(rand_permutation(j)) = num_bits_DOStdDither(rand_permutation(j)) + 1;
        end
        
        % Generate the random orthonormal frame
        rand_matrix = randn(N);
        [U, Sigma, V] = svd(rand_matrix);
        rand_orth = U*V';
        random_perm = randperm(N);
        S = rand_orth(random_perm(1:n),:);          % Columns constitute the tight frame
        
        % Compute Democratic (Kashin) representation
        K = 4*sqrt(log(1/mu))/(mu^2*sqrt(constant));    % Kashin's representation level
        
        a = zeros(N,1);                                 % Array to contain Kashin's coefficients
        M = norm(x,2)/sqrt(delta*N);                    % Initial truncation level

        max_iters = ceil(log(sqrt(N)/K)/log(1/eta));
        v = x;                                          % Initialize residual

        for j = 1:1:max_iters
            b = S'*v;                           % Compute frame representation of the residual
            b_hat = sign(b).*min(abs(b),M);     % Truncate frame coefficients
            v_recon = S*b_hat;                  % Reconstructed residual from truncated coefficients
            v = v - v_recon;                    % Update residual

            % Update Kasihn's coefficients and the truncation level
            a = a + sqrt(N).*b_hat;             
            M = eta*M;
        end

        x_coeff = a;                            % Coefficients of the Kashin representation
        
        % Standard dither quantization
        s_coeff = norm(x_coeff,Inf);                    % Dynamic range of the coefficients
        x_coeff_scaled = (1/s_coeff)*x_coeff;           % Normalize coefficients to unit dynamic range
        
        x_coeff_hat_DOStdDither = zeros(N,1);           % Initialize quantized coefficient vector
        
        for j = 1:1:N
    
            DELTA = 1/(2^num_bits_DOStdDither(j)-1);

            ind = floor(abs(x_coeff_scaled(j))/DELTA);
            lower_point = ind*DELTA;
            upper_point = (ind+1)*DELTA;

            p = (abs(x_coeff_scaled(j)) - lower_point)/DELTA;

            if(rand > p)
                x_coeff_hat_DOStdDither(j) = s_coeff*sign(x_coeff_scaled(j))*lower_point;
            else
                x_coeff_hat_DOStdDither(j) = s_coeff*sign(x_coeff_scaled(j))*upper_point;
            end

        end
        
        x_hat_DOStdDither = 1/sqrt(N)*S*x_coeff_hat_DOStdDither;              % Inverse transform
        
        % Compute normalized error for this realization and add it to the
        % cumulative sum to compute empirical average later
        alpha_DOStdDith = alpha_DOStdDith + norm(x_hat_DOStdDither - x,2)^2/norm(x,2)^2;
        
    end
    
    % Empirical normalized error
    alpha_array_DOStdDith(R_ind) = alpha_DOStdDith/num_realizations;
    
    % Effective number of bits per dimension
    effective_num_bits_per_dim_DO(R_ind) = R;
    
    % To track progress
    fprintf('Democratic (random orthonormal frame) + Standard dithering. Data rate = %f\n', R);
     
end


%%
% Top-k sparsification + Standard dither quantization

frac = 0.5;                                                                         % Fraction of the top magnitude coefficients to retain
k = floor(frac*n);
alpha_array_topk_StdDith = zeros(1, length(data_rate_variation));                   % To store normalized variance for each data rate
effective_num_bits_per_dim_topk_StdDith = zeros(1, length(data_rate_variation));    % Number of bits (including additional bits for sending indices etc.)

R_ind = 0;

for R = min_R:R_step:max_R
    
    R_ind = R_ind + 1;                          % Counter to increment data rate array indexing 
    
    alpha_topk_StdDith = 0;                     % Cumulative normalized error (to average out later)
    total_num_bits_topk = floor(n*R);
    
    for realiz_ind = 1:1:num_realizations
        
        % Generate a heavy-tailed random vector
        x = randn(n,1).^3;
        
        % Top-k sparsification step
        [x_sorted, indices] = sort(abs(x), 'descend');              % Sort the elements in ascending order of magnitude
        x_topk_trunc = x(indices(1:k));                             % A lower dimensional vector with only the top-k elements in descending order
        
        % Standard dither quantization of the k-dimensional truncation top-k vector
        % Allocating integral number of bits to each dimension
        num_bits_per_dim_topk = floor(total_num_bits_topk/k);
        num_bits_topk = num_bits_per_dim_topk*ones(1,k);                    % Initial allocation
        remaining_bits = total_num_bits_topk - k*num_bits_per_dim_topk;
        rand_permutation = randperm(k);
        for j = 1:1:remaining_bits
            num_bits_topk(rand_permutation(j)) = num_bits_topk(rand_permutation(j)) + 1;
        end
        
        s = norm(x_topk_trunc, Inf);                
        x_topk_trunc_scaled = (1/s)*x_topk_trunc;           % Normalize by the dynamic range
        x_topk_trunc_hat = zeros(k,1);                      % Initialize quantized truncated vector
        
        % Do dithering and scalar quantize each coordinate
        for j = 1:1:k
            
            DELTA = 1/(2^num_bits_topk(j)-1);
            
            ind = floor(abs(x_topk_trunc_scaled(j))/DELTA);
            lower_point = ind*DELTA;
            upper_point = (ind+1)*DELTA;
            
            p = (abs(x_topk_trunc_scaled(j)) - lower_point)/DELTA;
             
            if(rand > p)
                x_topk_trunc_hat(j) = s*sign(x_topk_trunc_scaled(j))*lower_point;
            else
                x_topk_trunc_hat(j) = s*sign(x_topk_trunc_scaled(j))*upper_point;
            end 
            
        end
        
        % Fill in the zeros
        x_hat_topk = zeros(n,1);
        for j = 1:1:k
            x_hat_topk(indices(j)) = x_topk_trunc_hat(j);
        end
        
        % Compute normalized error for this realization and add it to the
        % cumulative sum to compute empirical average later
        alpha_topk_StdDith = alpha_topk_StdDith + norm(x_hat_topk - x,2)^2/norm(x,2)^2;
        
    end
    
    % Empirical normalized error
    alpha_array_topk_StdDith(R_ind) = alpha_topk_StdDith/num_realizations;
    
    % Effective number of bits per dimension
    effective_num_bits_per_dim_topk_StdDith(R_ind) = (total_num_bits_topk + log2(nchoosek(n,k)))/n;
    
    % To track progress
    fprintf('Top-k + Standard dithering. Data rate = %f\n', R);
    
end

%%
% Top-k sparsification + Near-democratic (randomized Hadamard frame) + Standard dithering
frac = 0.5;                                                                           % Fraction of the top magnitude coefficients to retain
k = floor(frac*n);
Nk = 2^(ceil(log2(k)));                                                               % Higher dimension 
alpha_array_topk_NDStdDith = zeros(1, length(data_rate_variation));                   % To store normalized variance for each data rate
effective_num_bits_per_dim_topk_NDStdDith = zeros(1, length(data_rate_variation));    % Number of bits (including additional bits for sending indices etc.)

R_ind = 0;

for R = min_R:R_step:max_R
    
    R_ind = R_ind + 1;                          % Counter to increment data rate array indexing 
     
    alpha_topk_NDStdDith = 0;                   % Cumulative normalized variance (to average out later)
    total_num_bits_topk = floor(n*R);
    
    for realiz_ind = 1:1:num_realizations
        
        % Generate a heavy-tailed random vector
        x = randn(n,1).^3;
        
        % Top-k sparsification step
        [x_sorted, indices] = sort(abs(x), 'descend');              % Sort the elements in ascending order of magnitude
        x_topk_trunc = x(indices(1:k));                             % A lower dimensional vector with only the top-k elements in descending order
        
        % Near-democratic standard dither quantization of the k-dimensional truncation top-k vector
        % Allocating integral number of bits to each dimension
        num_bits_per_dim_topk_NDStdDither = floor(total_num_bits_topk/Nk);
        num_bits_topk_NDStdDither = num_bits_per_dim_topk_NDStdDither*ones(1,Nk);    % Initial allocation
        remaining_bits = total_num_bits_topk - Nk*num_bits_per_dim_topk_NDStdDither;
        rand_permutation = randperm(Nk);
        for j = 1:1:remaining_bits
            num_bits_topk_NDStdDither(rand_permutation(j)) = num_bits_topk_NDStdDither(rand_permutation(j)) + 1;
        end
        
        % Generate the randomized hadamard frame
        D = diag(2*(randi([0,1], Nk, 1) - 0.5));                        % Random diagonal matrix
        H = (1/sqrt(Nk))*hadamard(Nk);
        Id = eye(Nk);
        perm_rows = randperm(Nk);
        P = Id(perm_rows(1:k),:);                                       % Matrix for randomly selecting rows
        S = P*D*H;
        
        x_topk_trunc_coeff = S'*x_topk_trunc;                           % Near-democratic representation
        
        % Standard dither quantization
        s_coeff = norm(x_topk_trunc_coeff, Inf);                        % Dynamic range of the coefficients
        x_topk_trunc_coeff_scaled = (1/s_coeff)*x_topk_trunc_coeff;     % Normalize coefficients to unit dynamic range
        
        x_topk_trunc_coeff_hat_NDStdDither = zeros(Nk,1);               % Initialize quantized coefficient vector
        
        for j = 1:1:Nk
            
            DELTA = 1/(2^num_bits_topk_NDStdDither(j)-1);
            
            ind = floor(abs(x_topk_trunc_coeff_scaled(j))/DELTA);
            lower_point = ind*DELTA;
            upper_point = (ind+1)*DELTA;
            
            p = (abs(x_topk_trunc_coeff_scaled(j)) - lower_point)/DELTA;
            
            if(rand > p)
                x_topk_trunc_coeff_hat_NDStdDither(j) = s_coeff*sign(x_topk_trunc_coeff_scaled(j))*lower_point;
            else
                x_topk_trunc_coeff_hat_NDStdDither(j) = s_coeff*sign(x_topk_trunc_coeff_scaled(j))*upper_point;
            end
            
        end
        
        % Inverse transform
        x_topk_trunc_hat_NDStdDither = S*x_topk_trunc_coeff_hat_NDStdDither;
        
        % Fill in zeros
        x_hat_topk_NDStdDither = zeros(n,1);
        for j = 1:1:k
            x_hat_topk_NDStdDither(indices(j)) = x_topk_trunc_hat_NDStdDither(j);
        end
        
        alpha_topk_NDStdDith = alpha_topk_NDStdDith + norm(x_hat_topk_NDStdDither - x)^2/norm(x)^2;
        
    end
    
    % Average over different realizations
    alpha_array_topk_NDStdDith(R_ind) = alpha_topk_NDStdDith/num_realizations;
    
    % Effective number of bits per dimension
    effective_num_bits_per_dim_topk_NDStdDith(R_ind) = (total_num_bits_topk + log2(nchoosek(n,k)))/n;
    
    % To track progress
    fprintf('Top-k + Near-democratic(random Hadamard) + Standard dithering. Data rate = %f\n', R);
    
end


%%
% Top-k sparsification + Near-democratic (random orthonormal frame) + Standard dithering
frac = 0.5;                                                                             % Fraction of the top magnitude coefficients to retain
k = floor(frac*n);
lambda = 1;                                                                             % Aspect-ratio (redundancy) of the frame
Nk = ceil(lambda*k);                                                                    % Higher dimension  
alpha_array_topk_NDOStdDith = zeros(1, length(data_rate_variation));                    % To store normalized variance for each data rate
effective_num_bits_per_dim_topk_NDOStdDith = zeros(1, length(data_rate_variation));     % Number of bits (including additional bits for sending indices etc.)

R_ind = 0;

for R = min_R:R_step:max_R
    
    R_ind = R_ind + 1;                          % Counter to increment data rate array indexing 
     
    alpha_topk_NDOStdDith = 0;                  % Cumulative normalized variance (to average out later)
    total_num_bits_topk = floor(n*R);
    
    for realiz_ind = 1:1:num_realizations
        
        % Generate a heavy-tailed random vector
        x = randn(n,1).^3;
        
        % Top-k sparsification step
        [x_sorted, indices] = sort(abs(x), 'descend');              % Sort the elements in ascending order of magnitude
        x_topk_trunc = x(indices(1:k));                             % A lower dimensional vector with only the top-k elements in descending order
        
        % Near-democratic standard dither quantization of the k-dimensional truncation top-k vector
        % Allocating integral number of bits to each dimension
        num_bits_per_dim_topk_NDOStdDither = floor(total_num_bits_topk/Nk);
        num_bits_topk_NDOStdDither = num_bits_per_dim_topk_NDOStdDither*ones(1,Nk);    % Initial allocation
        remaining_bits = total_num_bits_topk - Nk*num_bits_per_dim_topk_NDOStdDither;
        rand_permutation = randperm(Nk);
        for j = 1:1:remaining_bits
            num_bits_topk_NDOStdDither(rand_permutation(j)) = num_bits_topk_NDOStdDither(rand_permutation(j)) + 1;
        end
        
        % Generate the randomized hadamard frame
        rand_matrix = randn(Nk);
        [U, Sigma, V] = svd(rand_matrix);
        rand_orth = U*V';
        random_perm = randperm(Nk);
        S = rand_orth(random_perm(1:k),:);          % Columns constitute the tight frame
        
        x_topk_trunc_coeff = S'*x_topk_trunc;       % Near-democratic representation
        
        % Standard dither quantization
        s_coeff = norm(x_topk_trunc_coeff, Inf);                        % Dynamic range of the coefficients
        x_topk_trunc_coeff_scaled = (1/s_coeff)*x_topk_trunc_coeff;     % Normalize coefficients to unit dynamic range
        
        x_topk_trunc_coeff_hat_NDOStdDither = zeros(Nk,1);               %Initialize quantized coefficient vector
        
        for j = 1:1:Nk
            
            DELTA = 1/(2^num_bits_topk_NDOStdDither(j)-1);
            
            ind = floor(abs(x_topk_trunc_coeff_scaled(j))/DELTA);
            lower_point = ind*DELTA;
            upper_point = (ind+1)*DELTA;
            
            p = (abs(x_topk_trunc_coeff_scaled(j)) - lower_point)/DELTA;
            
            if(rand > p)
                x_topk_trunc_coeff_hat_NDOStdDither(j) = s_coeff*sign(x_topk_trunc_coeff_scaled(j))*lower_point;
            else
                x_topk_trunc_coeff_hat_NDOStdDither(j) = s_coeff*sign(x_topk_trunc_coeff_scaled(j))*upper_point;
            end
            
        end
        
        % Inverse transform
        x_topk_trunc_hat_NDOStdDither = S*x_topk_trunc_coeff_hat_NDOStdDither;
        
        % Fill in zeros
        x_hat_topk_NDOStdDither = zeros(n,1);
        for j = 1:1:k
            x_hat_topk_NDOStdDither(indices(j)) = x_topk_trunc_hat_NDOStdDither(j);
        end
        
        alpha_topk_NDOStdDith = alpha_topk_NDOStdDith + norm(x_hat_topk_NDOStdDither - x)^2/norm(x)^2;
        
    end
    
    % Average over different realizations
    alpha_array_topk_NDOStdDith(R_ind) = alpha_topk_NDOStdDith/num_realizations;
    
    % Effective number of bits per dimension
    effective_num_bits_per_dim_topk_NDOStdDith(R_ind) = (total_num_bits_topk + log2(nchoosek(n,k)))/n;
    
    % To track progress
    fprintf('Top-k + Near-democratic(random orthonormal) + Standard dithering. Data rate = %f\n', R);
    
end


%%
% Top-k sparsification + Democratic (random orthonormal frame - Projected gradient descent) + Standard dithering
frac = 0.5;               % Fraction of the top magnitude coefficients to retain
k = floor(frac*n);

lambda = 1.8;
mu = lambda - 1;          % Parameter that controls the redundancy of the frame
constant = 0.3;           % Trades off between UP parameters and probability that the random frame satisfies UP

% Uncertainty principle (UP) parameters
eta = 1 - 0.25*mu;
delta = constant*mu^2/log(1/mu);

Nk = ceil(lambda*k);           % Dimension of the higher space
                                                                 
alpha_array_topk_DOStdDith = zeros(1, length(data_rate_variation));                     % To store normalized variance for each data rate
effective_num_bits_per_dim_topk_DOStdDith = zeros(1, length(data_rate_variation));      % Number of bits (including additional bits for sending indices etc.)

R_ind = 0;

for R = min_R:R_step:max_R
    
    R_ind = R_ind + 1;                          % Counter to increment data rate array indexing 
     
    alpha_topk_DOStdDith = 0;                   % Cumulative normalized error (to average out later)
    total_num_bits_topk = floor(n*R);
    
    for realiz_ind = 1:1:num_realizations
        
        % Generate a heavy-tailed random vector
        x = randn(n,1).^3;
        
        % Top-k sparsification step
        [x_sorted, indices] = sort(abs(x), 'descend');              % Sort the elements in ascending order of magnitude
        x_topk_trunc = x(indices(1:k));                             % A lower dimensional vector with only the top-k elements in descending order
        
        % Near-democratic standard dither quantization of the k-dimensional truncation top-k vector
        % Allocating integer number of bits to each dimension
        num_bits_per_dim_topk_DOStdDither = floor(total_num_bits_topk/Nk);
        num_bits_topk_DOStdDither = num_bits_per_dim_topk_DOStdDither*ones(1,Nk);    % Initial allocation
        remaining_bits = total_num_bits_topk - Nk*num_bits_per_dim_topk_DOStdDither;
        rand_permutation = randperm(Nk);
        for j = 1:1:remaining_bits
            num_bits_topk_DOStdDither(rand_permutation(j)) = num_bits_topk_DOStdDither(rand_permutation(j)) + 1;
        end
        
        % Generate the random orthonormal frame
        rand_matrix = randn(Nk);
        [U, Sigma, V] = svd(rand_matrix);
        rand_orth = U*V';
        random_perm = randperm(Nk);
        S = rand_orth(random_perm(1:k),:);          % Columns constitute the tight frame
        
        % Compute Democratic (Kashin) representation
        K = 4*sqrt(log(1/mu))/(mu^2*sqrt(constant));     % Kashin's representation level
        
        a = zeros(Nk,1);                                 % Array to contain Kashin's coefficients
        M = norm(x_topk_trunc,2)/sqrt(delta*Nk);         %Initial truncation level

        max_iters = ceil(log(sqrt(Nk)/K)/log(1/eta));
        v = x_topk_trunc;                                % Initialize residual

        for j = 1:1:max_iters
            b = S'*v;                           % Compute frame representation of the residual
            b_hat = sign(b).*min(abs(b),M);     % Truncate frame coefficients
            v_recon = S*b_hat;                  % Reconstructed residual from truncated coefficients
            v = v - v_recon;                    % Update residual

            % Update Kasihn's coefficients and the truncation level
            a = a + sqrt(Nk).*b_hat;             
            M = eta*M;
        end

        % Coefficients of the Kashin representation
        x_topk_trunc_coeff = a;                            

        % Standard dither quantization
        s_coeff = norm(x_topk_trunc_coeff, Inf);                        % Dynamic range of the coefficients
        x_topk_trunc_coeff_scaled = (1/s_coeff)*x_topk_trunc_coeff;     % Normalize coefficients to unit dynamic range
        
        x_topk_trunc_coeff_hat_DOStdDither = zeros(Nk,1);               % Initialize quantized coefficient vector
        
        for j = 1:1:Nk
            
            DELTA = 1/(2^num_bits_topk_DOStdDither(j)-1);
            
            ind = floor(abs(x_topk_trunc_coeff_scaled(j))/DELTA);
            lower_point = ind*DELTA;
            upper_point = (ind+1)*DELTA;
            
            p = (abs(x_topk_trunc_coeff_scaled(j)) - lower_point)/DELTA;
            
            if(rand > p)
                x_topk_trunc_coeff_hat_DOStdDither(j) = s_coeff*sign(x_topk_trunc_coeff_scaled(j))*lower_point;
            else
                x_topk_trunc_coeff_hat_DOStdDither(j) = s_coeff*sign(x_topk_trunc_coeff_scaled(j))*upper_point;
            end
            
        end
        
        % Inverse transform
        x_topk_trunc_hat_DOStdDither = (1/sqrt(Nk))*S*x_topk_trunc_coeff_hat_DOStdDither;
        
        % Fill in zeros
        x_hat_topk_DOStdDither = zeros(n,1);
        for j = 1:1:k
            x_hat_topk_DOStdDither(indices(j)) = x_topk_trunc_hat_DOStdDither(j);
        end
        
        alpha_topk_DOStdDith = alpha_topk_DOStdDith + norm(x_hat_topk_DOStdDither - x)^2/norm(x)^2;
        
    end
    
    % Average over different realizations
    alpha_array_topk_DOStdDith(R_ind) = alpha_topk_DOStdDith/num_realizations;
    
    % Effective number of bits per dimension
    effective_num_bits_per_dim_topk_DOStdDith(R_ind) = (total_num_bits_topk + log2(nchoosek(n,k)))/n;
    
    % To track progress
    fprintf('Top-k + Democratic(random orthonormal) + Standard dithering. Data rate = %f\n', R);
    
end

%%
% Plot results

figure;
plot(effective_num_bits_per_dim, alpha_array_StdDith, '-', 'LineWidth', 1.5, 'Color', 'r');
hold on;
plot(effective_num_bits_per_dim_ND, alpha_array_NDStdDith, '-', 'LineWidth', 1.5, 'Color', 'b');
plot(effective_num_bits_per_dim_NDO, alpha_array_NDOStdDith, '-', 'LineWidth', 1.5, 'Color', 'g');
plot(effective_num_bits_per_dim_DO, alpha_array_DOStdDith, '-', 'LineWidth', 1.5, 'Color', 'k');
plot(effective_num_bits_per_dim_topk_StdDith, alpha_array_topk_StdDith, '--', 'LineWidth', 1.5, 'Color', 'r');
plot(effective_num_bits_per_dim_topk_NDStdDith, alpha_array_topk_NDStdDith, '--', 'LineWidth', 1.5, 'Color', 'b');
plot(effective_num_bits_per_dim_topk_NDOStdDith, alpha_array_topk_NDOStdDith, '--', 'LineWidth', 1.5, 'Color', 'g');
plot(effective_num_bits_per_dim_topk_DOStdDith, alpha_array_topk_DOStdDith, '--', 'LineWidth', 1.5, 'Color', 'k');
xlabel('Effective number of bits per dimension');
ylabel('Normalized variance');
title('Map of compression methods');
grid on;
legend('Vanilla standard dither', 'Near-democratic (randomized Hadamard) + Standard dither', 'Near-democratic (random orthonormal) + Standard dither', 'Democratic (random orthonormal) + Standard dither', 'Top-k + Standard dither', 'Top-k + Near-democratic (randomized Hadamard) + Standard dither', 'Top-k + Near-democratic (random orthonormal) + Standard dither', 'Top-k + Democratic (random orthonormal) + Standard dither');

%%
% Save workspace

filename = 'Compression_methods_map_workspace.mat'
save(filename)
