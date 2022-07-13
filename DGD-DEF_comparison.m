% This code implements Distributed Gradient Descent with Democratically Encoded Feedback 
% (DGD-DEF) for a least squares problem in the single server and single worker setting 
% where the communication link between them is subject to a data rate-constraint of R 
% bits per dimension.

close all;
clear all;
clc;

num_realizations = 5;          % No. of ensemble realizations
m = 500;                       % No. of data points
n = 116;                       % Dimension of the problem
T = 60;                        % Time horizon

% Parameters of the Hadamard dictionary 
r = 1.1;                         % Minimum redundancy of the dictionary for hadamard heuristic
N = 2^(ceil(log2(r*n)));         % Dimension of the higher dimensional space
redundancy = N/n;                % True redundancy 

% Parameters of the random orthonormal dictionary for near-democratic coding
lambda_ndo = 1.1;                             % Aspect-ratio (redundancy) of the frame
N_ndo = ceil(lambda_ndo*n);                   % Higher dimension

% Parameters of the random orthonormal dictionary for democratic coding
lambda_do = 1.2;                              % Aspect-ratio (redundancy) of the frame
N_do = ceil(lambda_do*n);                     % Higher dimension

% Data rate (per dimension) variation range
min_R = 0.1;
R_step = 0.2;
max_R = 5;
data_rate_variation_length = floor((max_R - min_R)/R_step) + 1;
R_ind = 0;                      % Used for indexing the array

% Quantizer design parameters
% Need a separate array for this because if R is fractional, total number of bits might not be a
% perfect multiple of n. But number of bits alloted to each dimension must be an integer.
num_bits = zeros(1,n);              % Number of bits allocated to each dimension
num_bits_hadamard = zeros(1,N);     % No. of bits allocated to each dimension in the higher dimensional space
num_bits_ndo = zeros(1,N_ndo);      
num_bits_do = zeros(1, N_do);       

% To compute the empiricial convergence rate
conv_rate_DQGD_array = zeros(1, data_rate_variation_length);
conv_rate_DQGD_norm_separate_array = zeros(1, data_rate_variation_length);
conv_rate_unquantized_gd_array = zeros(1, data_rate_variation_length);
conv_rate_DQGD_hadamard_array = zeros(1, data_rate_variation_length);
conv_rate_DQGD_HadamardDemocratic_array = zeros(1, data_rate_variation_length);
conv_rate_DQGD_ndo_array = zeros(1, data_rate_variation_length);
conv_rate_DQGD_do_array = zeros(1, data_rate_variation_length);

% Generate synthetic dataset only once
A = randn(m,n).^5;              % Data matrix
y = randn(m,1);                 % Output values

% Compute problem parameters
eigenval = eig(A'*A);
L = max(eigenval);          % Smoothness constant          
mu = min(eigenval);         % Strong-convexity constant

% Optimal solution in closed form
x_star = A\y;

% Initializing unquantized GD
x_init = randn(n,1);
D = norm(x_init - x_star, 2);             % Initial distance to optimality

% Loop over different data rates
for R = min_R:R_step:max_R
    
    % To maintain a counter for array indexing
    R_ind = R_ind + 1;
    
    % Quantizer design for unit dynamic range
    total_num_bits = floor(n*R);
    
    % Allocation of total_num_bits to each dimension (for vanilla scalar quantizer)
    num_bits_per_dim = floor(total_num_bits/n);             
    num_bits = num_bits_per_dim*ones(1,n);                      % Initial allocation
    remaining_bits = total_num_bits - n*num_bits_per_dim;
    rand_permutation = randperm(n);
    for j = 1:1:remaining_bits
        num_bits(rand_permutation(j)) = num_bits(rand_permutation(j)) + 1;
    end

    num_points_per_dim = 2.^num_bits;          % Number of points per dimension
    rho = sqrt(n);                             % Covering efficiency of scalar quantizer             
    
    
    % Allocation of total_num_bits to each dimension for near-democratic coding with randomized Hadamard frame
    num_bits_per_dim_hadamard = floor(total_num_bits/N);
    num_bits_hadamard = num_bits_per_dim_hadamard*ones(1,N);         % Initial allocation
    remaining_bits = total_num_bits - N*num_bits_per_dim_hadamard;
    rand_permutation = randperm(N);
    for j = 1:1:remaining_bits
        num_bits_hadamard(rand_permutation(j)) = num_bits_hadamard(rand_permutation(j)) + 1;
    end
    
    num_points_per_dim_hadamard = 2.^num_bits_hadamard;     % No. of points per dimension for hadamard heuristic
    
    
    % Allocation of total_num_bits to each dimension for near-democratic coding with random orthonormal frame
    num_bits_per_dim_ndo = floor(total_num_bits/N_ndo);
    num_bits_ndo = num_bits_per_dim_ndo*ones(1,N_ndo);              % Initial allocation
    remaining_bits = total_num_bits - N_ndo*num_bits_per_dim_ndo;
    rand_permutation = randperm(N_ndo);
    for j = 1:1:remaining_bits
        num_bits_ndo(rand_permutation(j)) = num_bits_ndo(rand_permutation(j)) + 1;
    end
    
    num_points_per_dim_ndo = 2.^num_bits_ndo;          % No. of points per dimension for near-democratic random orthonormal frame
    
    
    % Allocation of total_num_bits to each dimension for democratic coding with random orthonormal frame
    num_bits_per_dim_do = floor(total_num_bits/N_do);
    num_bits_do = num_bits_per_dim_do*ones(1,N_do);              % Initial allocation
    remaining_bits = total_num_bits - N_do*num_bits_per_dim_do;
    rand_permutation = randperm(N_do);
    for j = 1:1:remaining_bits
        num_bits_do(rand_permutation(j)) = num_bits_do(rand_permutation(j)) + 1;
    end
    
    num_points_per_dim_do = 2.^num_bits_do;          % No. of points per dimension for near-democratic random orthonormal frame
    
    
    % Length of interval in each dimension for scalar quantizer with dynamic range = 1
    resolution_per_dim = 2./num_points_per_dim;                         % Vanilla
    resolution_per_dim_hadamard = 2./num_points_per_dim_hadamard;       % Hadamard heuristic
    resolution_per_dim_ndo = 2./num_points_per_dim_ndo;                 % Near-democratic random orthonormal frame
    resolution_per_dim_do = 2./num_points_per_dim_do;                   % Democratic random orthonormal frame

    
    % To compute the ensemble average of the empirical convergence rate
    conv_rate_unquantized_gd_sum = 0;
    conv_rate_DQGD_sum = 0;
    conv_rate_DQGD_norm_separate_sum = 0;
    conv_rate_DQGD_hadamard_sum = 0;
    conv_rate_DQGD_HadamardDemocratic_sum = 0;
    conv_rate_DQGD_ndo_sum = 0;
    conv_rate_DQGD_do_sum = 0;
    
    % Loop over different realizations
    for realiz_ind = 1:1:num_realizations

        %Initializing unquantized GD
        %x_init = randn(n,1);
        %D = norm(x_init - x_star, 2);             %Initial distance to optimality

        %----------------------------------------------------------------------
        % Unquantized gradient descent
        x = x_init;
        eta = 2/(L + mu);               % Constant (optimal) step size
        %eta = 1e-5;
        
        for i = 1:1:T
            % Compute gradient
            grad = A'*(A*x - y);

            % Take a descent step
            x = x - eta*grad;     
        end

        % Final distance to optimality
        final_DistToOpt = norm(x - x_star, 2); 

        % Convergence rate
        conv_rate_unquantized_gd = (final_DistToOpt/D)^(1/T);
        conv_rate_unquantized_gd_sum = conv_rate_unquantized_gd_sum + conv_rate_unquantized_gd;

        %----------------------------------------------------------------------
        % Differentially quantized gradient descent (vanilla scalar quantizer with predetermined dynamic ranges)
        x_hat = x_init;                             % DQGD iterate
        e = zeros(n,1);                             % Error in previous time step
        eta = 2/(L + mu);                           % Constant stepsize
        eta_star = 2/(L + mu);                      % Optimal stepsize
        nu_sub_eta = sqrt(1 - eta_star*L*mu*eta);   % Unquantized GD convergence rate (with this particular step size choice)
        
        for i = 1:1:T
            
            % Get point on unquantized GD trajectory
            z = x_hat + eta*e;
            
            % Compute gradient on unquantized GD trajectory
            grad = A'*(A*z - y);
            
            % Compute input to quantizer
            u = grad - e;
            
            % Do scalar quantization over different dimensions
            q_scaled = zeros(n,1);         % Initializing the quantized vector
            ratio = (2^R)/rho;             % Ratio of subsequent terms in geometric sum
            r_i = ratio^(-i)*(1 - (nu_sub_eta*ratio)^(i+1))/(1-nu_sub_eta*ratio)*L*D;   % Scaled dynamic range
            u_scaled = (1/r_i)*u;          % Scale the input to the quantizer to lie within unit dynamic range
            
            % Quantize each dimension of scaled input with scalar quantizer of unit dynamic range
            for j = 1:1:n
                
                % Get the first and last quantization points
                first_quant_point = -1 + resolution_per_dim(j)/2;
                last_quant_point = 1 - resolution_per_dim(j)/2;
                
                % Quantize
                if (num_bits(j) == 0)
                    q_scaled(j) = 0;                           % If no bits are allocated to a particular dimension simply send 0 for that coordinate
                
                elseif (u_scaled(j) <= first_quant_point)
                    q_scaled(j) = first_quant_point;           % Check if lies between -1 and first quantization point
                    
                elseif (u_scaled(j) >= last_quant_point)
                    q_scaled(j) = last_quant_point;            % Check if lies between last quantization point and +1
                
                % Otherwise lies between first_quant_point and last_quant_point
                else
                    
                    lower_quant_idx = floor((u_scaled(j)-first_quant_point)/resolution_per_dim(j));     %Index of lower quantization point
                    lower_quant_point = first_quant_point + lower_quant_idx*resolution_per_dim(j);                     %Lower quantization point
                    upper_quant_point = first_quant_point + (lower_quant_idx+1)*resolution_per_dim(j);                 %Upper quantization point

                    % Do nearest neighbor quantization
                    if ((u_scaled(j) - lower_quant_point) <= resolution_per_dim(j)/2)
                        q_scaled(j) = lower_quant_point;
                    else
                        q_scaled(j) = upper_quant_point;
                    end
                    
                end                         
            end
            
            q = r_i*q_scaled;       % Scale back the quantized vector
            
            % Update error (to be fed back)
            e = q - u;
            
            % Server takes a descent step and updates iterate
            x_hat = x_hat - eta*q;
            
        end
        
        % Final distance to optimality
        final_DistToOpt = norm(x_hat - x_star, 2);
        
        % Convergence rate
        conv_rate_DQGD = (final_DistToOpt/D)^(1/T);
        conv_rate_DQGD_sum = conv_rate_DQGD_sum + conv_rate_DQGD;
        
        %------------------------------------------------------------------
        % Differentially Quantized Gradient Descent (vanilla scalar quantizer with l_infty norm transmitted separately)
        x_hat = x_init;                             % DQGD iterate (hadamard heuristic)
        e = zeros(n,1);                             % Error in the previous time step
        eta = 2/(L + mu);                           % Constant stepsize choice
        eta_star = 2/(L + mu);                      % Optimal stepsize
        nu_sub_eta = sqrt(1 - eta_star*L*mu*eta);   % Unquantized GD convergence rate (with optimal step size choice)
        
        for i = 1:1:T
            
            % Get point on unquantized GD trajectory
            z = x_hat + eta*e;
            
            % Compute gradient on unquantized GD trajectory
            grad = A'*(A*z - y);
            
            % Compute input to quantizer
            u = grad - e;
            
            % Do scalar quantization of the vector directly
            quant_scaled = zeros(n,1);            % Initializing the quantized coefficients
            
            dyn_range = norm(u,inf);
            u_scaled_dyn = 1/dyn_range*u;
            
            % Quantize each dimension of the scaled input with scalar quantizer of unit dynamic range
            for j = 1:1:n
                
                % Get the first and last quantization points
                first_quant_point = -1 + resolution_per_dim(j)/2;
                last_quant_point = 1 - resolution_per_dim(j)/2;
                
                % Quantize
                if (num_bits(j) == 0)
                    quant_scaled(j) = 0;
                    
                elseif (u_scaled_dyn(j) <= first_quant_point)
                    quant_scaled(j) = first_quant_point;           % Check if lies between -1 and first quantization point
                    
                elseif (u_scaled_dyn(j) >= last_quant_point)
                    quant_scaled(j) = last_quant_point;            % Check if lies between last quantization point and +1
                    
                % Otherwise lies between first_quant_point and last_quant_point
                else
                    
                    lower_quant_idx = floor((u_scaled_dyn(j)-first_quant_point)/resolution_per_dim(j));     %Index of lower quantization point
                    lower_quant_point = first_quant_point + lower_quant_idx*resolution_per_dim(j);                     %Lower quantization point
                    upper_quant_point = first_quant_point + (lower_quant_idx+1)*resolution_per_dim(j);                 %Upper quantization point

                    % Do nearest neighbor quantization
                    if ((u_scaled_dyn(j) - lower_quant_point) <= resolution_per_dim(j)/2)
                        quant_scaled(j) = lower_quant_point;
                    else
                        quant_scaled(j) = upper_quant_point;
                    end
                    
                end   
            end
            
            q = quant_scaled*dyn_range;
                       
            % Update error (to be fed back)
            e = q - u;
            
            % Server takes a descent step and updates iterate
            x_hat = x_hat - eta*q;
            
        end
        
        % Final distance to optimality
        final_DistToOpt = norm(x_hat - x_star, 2);
        
        % Convergence rate
        conv_rate_DQGD_norm_separate = (final_DistToOpt/D)^(1/T);
        conv_rate_DQGD_norm_separate_sum = conv_rate_DQGD_norm_separate_sum + conv_rate_DQGD_norm_separate;
        
        
        %------------------------------------------------------------------
        % DGD-DEF (near-democratic coding with randomized Hadamard frame)
        x_hat_h = x_init;                             % Iterate (near-democratic Hadamard)
        e_h = zeros(n,1);                             % Error in the previous time step
        eta_h = 2/(L + mu);                           % Constant stepsize choice
        eta_star = 2/(L + mu);                        % Optimal stepsize
        nu_sub_eta_h = sqrt(1 - eta_star*L*mu*eta_h); % Unquantized GD convergence rate (with optimal step size choice)
        
        for i = 1:1:T
            
            % Get point on unquantized GD trajectory
            z_h = x_hat_h + eta_h*e_h;
            
            % Compute gradient on unquantized GD trajectory
            grad_h = A'*(A*z_h - y);
            
            % Compute input to quantizer
            u_h = grad_h - e_h;
            
            % Get randomized hadamard transform of u_h (input to the quantizer)
            D2 = diag(2*(randi([0,1], N, 1) - 0.5));        % Random diagonal matrix
            H = (1/sqrt(N))*hadamard(N);
            Id = eye(N);
            perm_rows = randperm(N);
            P = Id(perm_rows(1:n),:);                       % Matrix for randomly selecting rows
            D1 = diag(2*(randi([0,1], n, 1) - 0.5));        % Pre-multiplication random diagonal matrix
            S = P*D2*H;
            hadamard_coeff = S'*u_h;
            
            % Do scalar quantization of the hadamard coefficients
            quant_coeff_scaled = zeros(N,1);            % Initializing the quantized coefficients
            u_h_scaled = hadamard_coeff;
            
            dyn_range = norm(u_h_scaled,inf);
            u_h_scaled_dyn = 1/dyn_range*u_h_scaled;
            
            % Quantize each dimension of the scaled input with scalar quantizer of unit dynamic range
            for j = 1:1:N
                
                % Get the first and last quantization points
                first_quant_point = -1 + resolution_per_dim_hadamard(j)/2;
                last_quant_point = 1 - resolution_per_dim_hadamard(j)/2;
                
                % Quantize
                if (num_bits_hadamard(j) == 0)
                    quant_coeff_scaled(j) = 0;
                    
                elseif (u_h_scaled_dyn(j) <= first_quant_point)
                    quant_coeff_scaled(j) = first_quant_point;           %Check if lies between -1 and first quantization point
                    
                elseif (u_h_scaled_dyn(j) >= last_quant_point)
                    quant_coeff_scaled(j) = last_quant_point;           %Check if lies between -1 and first quantization point
                    
                % Otherwise lies between first_quant_point and last_quant_point
                else
                    
                    lower_quant_idx = floor((u_h_scaled_dyn(j)-first_quant_point)/resolution_per_dim_hadamard(j));     %Index of lower quantization point
                    lower_quant_point = first_quant_point + lower_quant_idx*resolution_per_dim_hadamard(j);                     %Lower quantization point
                    upper_quant_point = first_quant_point + (lower_quant_idx+1)*resolution_per_dim_hadamard(j);                 %Upper quantization point

                    % Do nearest neighbor quantization
                    if ((u_h_scaled_dyn(j) - lower_quant_point) <= resolution_per_dim_hadamard(j)/2)
                        quant_coeff_scaled(j) = lower_quant_point;
                    else
                        quant_coeff_scaled(j) = upper_quant_point;
                    end
                    
                end   
            end
            
            quant_coeff_scaled = quant_coeff_scaled*dyn_range;
            quant_coeff = quant_coeff_scaled;
            q_h = S*quant_coeff;                          % Inverse transform from quantized coefficients
            
            % Update error (to be fed back)
            e_h = q_h - u_h;
            
            % Server takes a descent step and updates iterate
            x_hat_h = x_hat_h - eta_h*q_h;
            
        end
        
        % Final distance to optimality
        final_DistToOpt_h = norm(x_hat_h - x_star, 2);
        
        % Convergence rate
        conv_rate_DQGD_hadamard = (final_DistToOpt_h/D)^(1/T);
        conv_rate_DQGD_hadamard_sum = conv_rate_DQGD_hadamard_sum + conv_rate_DQGD_hadamard;
        
        
        %------------------------------------------------------------------
        %DGD-DEF (near-democratic coding with random orthonormal frame)
        x_hat_ndo = x_init;                                 % Iterate (near-democratic Hadamard)
        e_ndo = zeros(n,1);                                 % Error in the previous time step
        eta_ndo = 2/(L + mu);                               % Constant stepsize choice
        eta_star = 2/(L + mu);                              % Optimal stepsize
        nu_sub_eta_ndo = sqrt(1 - eta_star*L*mu*eta_ndo);   % Unquantized GD convergence rate (with optimal step size choice)
        
        for i = 1:1:T
            
            % Get point on unquantized GD trajectory
            z_ndo = x_hat_ndo + eta_ndo*e_ndo;
            
            % Compute gradient on unquantized GD trajectory
            grad_ndo = A'*(A*z_ndo - y);
            
            % Compute input to quantizer
            u_ndo = grad_ndo - e_ndo;
            
            % Get randomized hadamard transform of u_ndo (input to the quantizer)
            rand_matrix = randn(N_ndo);
            [U, Sigma, V] = svd(rand_matrix);
            rand_orth = U*V';
            random_perm = randperm(N_ndo);
            S = rand_orth(random_perm(1:n),:);          % Columns constitute the tight frame 
            
            % Get the near-democratic repesentation
            u_ndo_coeff = S'*u_ndo;
            
            % Do scalar quantization of the hadamard coefficients
            quant_coeff_scaled = zeros(N_ndo,1);        % Initializing the quantized coefficients
            
            dyn_range = norm(u_ndo_coeff,inf);
            u_ndo_scaled_dyn = 1/dyn_range*u_ndo_coeff;     %Scale it to unit dynamic range
            
            % Quantize each dimension of the scaled input with scalar quantizer of unit dynamic range
            for j = 1:1:N_ndo
                
                % Get the first and last quantization points
                first_quant_point = -1 + resolution_per_dim_ndo(j)/2;
                last_quant_point = 1 - resolution_per_dim_ndo(j)/2;
                
                % Quantize
                if (num_bits_ndo(j) == 0)
                    quant_coeff_scaled(j) = 0;
                    
                elseif (u_ndo_scaled_dyn(j) <= first_quant_point)
                    quant_coeff_scaled(j) = first_quant_point;          % Check if lies between -1 and first quantization point
                    
                elseif (u_ndo_scaled_dyn(j) >= last_quant_point)
                    quant_coeff_scaled(j) = last_quant_point;           % Check if lies between -1 and first quantization point
                    
                % Otherwise lies between first_quant_point and last_quant_point
                else
                    
                    lower_quant_idx = floor((u_ndo_scaled_dyn(j)-first_quant_point)/resolution_per_dim_ndo(j));     %Index of lower quantization point
                    lower_quant_point = first_quant_point + lower_quant_idx*resolution_per_dim_ndo(j);                     %Lower quantization point
                    upper_quant_point = first_quant_point + (lower_quant_idx+1)*resolution_per_dim_ndo(j);                 %Upper quantization point

                    % Do nearest neighbor quantization
                    if ((u_ndo_scaled_dyn(j) - lower_quant_point) <= resolution_per_dim_ndo(j)/2)
                        quant_coeff_scaled(j) = lower_quant_point;
                    else
                        quant_coeff_scaled(j) = upper_quant_point;
                    end
                    
                end   
            end
            
            quant_coeff = quant_coeff_scaled*dyn_range;
            q_ndo = S*quant_coeff;                          % Inverse transform from quantized coefficients
            
            % Update error (to be fed back)
            e_ndo = q_ndo - u_ndo;
            
            % Server takes a descent step and updates iterate
            x_hat_ndo = x_hat_ndo - eta_ndo*q_ndo;
            
        end
        
        % Final distance to optimality
        final_DistToOpt_ndo = norm(x_hat_ndo - x_star, 2);
        
        % Convergence rate
        conv_rate_DQGD_ndo = (final_DistToOpt_ndo/D)^(1/T);
        conv_rate_DQGD_ndo_sum = conv_rate_DQGD_ndo_sum + conv_rate_DQGD_ndo;
        
        
        %------------------------------------------------------------------
        %DGD-DEF (democratic coding with randomized Hadamard frame)
        x_hat_hd = x_init;                              % Iterate (hadamard heuristic)
        e_hd = zeros(n,1);                              % Error in the previous time step
        eta_hd = 2/(L + mu);                            % Constant stepsize choice  
        eta_star = 2/(L + mu);                          % Optimal stepsize
        nu_sub_eta_hd = sqrt(1 - eta_star*L*mu*eta_hd); % Unquantized GD convergence rate (with optimal step size choice)
        
        for i = 1:1:T
            
            % Get point on unquantized GD trajectory
            z_hd = x_hat_hd + eta_hd*e_hd;
            
            % Compute gradient on unquantized GD trajectory
            grad_hd = A'*(A*z_hd - y);
            
            % Compute input to quantizer
            u_hd = grad_hd - e_hd;
            
            % Get randomized hadamard transform of u_h (input to the quantizer)
            D2 = diag(2*(randi([0,1], N, 1) - 0.5));        % Random diagonal matrix
            H = (1/sqrt(N))*hadamard(N);
            Id = eye(N);
            perm_rows = randperm(N);
            P = Id(perm_rows(1:n),:);                       % Matrix for randomly selecting rows
            D1 = diag(2*(randi([0,1], n, 1) - 0.5));        % Pre-multiplication random diagonal matrix
            S = P*D2*H;
            
            % Normalize the quantity to be encoded to unit norm
            u_hd_s = u_hd;
            
            % Compute democratic representation
            cvx_begin quiet
                variable hd_coefficient(N)
                minimize norm(hd_coefficient,Inf)
                subject to 
                    u_hd_s == S*hd_coefficient;
            cvx_end
            hdcoeff = hd_coefficient;
            
            % Do scalar quantization of the hadamard coefficients
            quant_hdcoeff_scaled = zeros(N,1);            % Initializing the quantized coefficients
            hd_coeff_scaled = hdcoeff;
            
            dyn_range = norm(hd_coeff_scaled, inf);
            hd_coeff_scaled = 1/dyn_range*hd_coeff_scaled;
            
            % Quantize each dimension of the scaled input with scalar quantizer of unit dynamic range
            for j = 1:1:N
                
                % Get the first and last quantization points
                first_quant_point = -1 + resolution_per_dim_hadamard(j)/2;
                last_quant_point = 1 - resolution_per_dim_hadamard(j)/2;
                
                % Quantize
                if (num_bits_hadamard(j) == 0)
                    quant_hdcoeff_scaled(j) = 0;
                    
                elseif (hd_coeff_scaled(j) <= first_quant_point)
                    quant_hdcoeff_scaled(j) = first_quant_point;          % Check if lies between -1 and first quantization point
                    
                elseif (hd_coeff_scaled(j) >= last_quant_point)
                    quant_hdcoeff_scaled(j) = last_quant_point;           % Check if lies between -1 and first quantization point
                    
                % Otherwise lies between first_quant_point and last_quant_point
                else
                    
                    lower_quant_idx = floor((hd_coeff_scaled(j)-first_quant_point)/resolution_per_dim_hadamard(j));     %Index of lower quantization point
                    lower_quant_point = first_quant_point + lower_quant_idx*resolution_per_dim_hadamard(j);                     %Lower quantization point
                    upper_quant_point = first_quant_point + (lower_quant_idx+1)*resolution_per_dim_hadamard(j);                 %Upper quantization point

                    % Do nearest neighbor quantization
                    if ((hd_coeff_scaled(j) - lower_quant_point) <= resolution_per_dim_hadamard(j)/2)
                        quant_hdcoeff_scaled(j) = lower_quant_point;
                    else
                        quant_hdcoeff_scaled(j) = upper_quant_point;
                    end
                    
                end   
            end
            
            quant_hdcoeff_scaled = quant_hdcoeff_scaled*dyn_range;
            quant_hdcoeff = quant_hdcoeff_scaled;       % Scale back the quantized vector
            q_hd = S*quant_hdcoeff;
            
            
            % Update error (to be fed back)
            e_hd = q_hd - u_hd;
            
            % Server takes a descent step and updates iterate
            x_hat_hd = x_hat_hd - eta_hd*q_hd;
            
            % To track progress
            fprintf('Rate: %f, Realization: %d, Iteration: %d\n', R, realiz_ind, i);
            
        end
        
        % Final distance to optimality
        final_DistToOpt_hd = norm(x_hat_hd - x_star, 2);
        
        % Convergence rate
        conv_rate_DQGD_HadamardDemocratic = (final_DistToOpt_hd/D)^(1/T);
        conv_rate_DQGD_HadamardDemocratic_sum = conv_rate_DQGD_HadamardDemocratic_sum + conv_rate_DQGD_HadamardDemocratic;
        
        
        %------------------------------------------------------------------
        % DGD-DEF (Democratic coding with random orthonormal frame)
        x_hat_do = x_init;                              % Iterate (Democratic random orthonormal)
        e_do = zeros(n,1);                              % Error in the previous time step
        eta_do = 2/(L + mu);                            % Constant stepsize choice
        eta_star = 2/(L + mu);                          % Optimal stepsize
        nu_sub_eta_do = sqrt(1 - eta_star*L*mu*eta_do); % Unquantized GD convergence rate (with optimal step size choice)
        
        for i = 1:1:T
            
            % Get point on unquantized GD trajectory
            z_do = x_hat_do + eta_do*e_do;
            
            % Compute gradient on unquantized GD trajectory
            grad_do = A'*(A*z_do - y);
            
            % Compute input to quantizer
            u_do = grad_do - e_do;
            
            % Get randomized hadamard transform of u_ndo (input to the quantizer)
            rand_matrix = randn(N_do);
            [U, Sigma, V] = svd(rand_matrix);
            rand_orth = U*V';
            random_perm = randperm(N_do);
            S = rand_orth(random_perm(1:n),:);          % Columns constitute the tight frame 
            
            % Get the democratic repesentation
            cvx_begin quiet
                variable do_coefficient(N_do)
                minimize norm(do_coefficient,Inf)
                subject to 
                    u_do == S*do_coefficient;
            cvx_end
            do_coeff = do_coefficient;
            
            % Do scalar quantization of the hadamard coefficients
            quant_coeff_scaled = zeros(N_do,1);             % Initializing the quantized coefficients
            
            dyn_range = norm(do_coeff,inf);
            do_coeff_scaled_dyn = 1/dyn_range*do_coeff;     % Scale it to unit dynamic range
            
            % Quantize each dimension of the scaled input with scalar quantizer of unit dynamic range
            for j = 1:1:N_do
                
                % Get the first and last quantization points
                first_quant_point = -1 + resolution_per_dim_do(j)/2;
                last_quant_point = 1 - resolution_per_dim_do(j)/2;
                
                % Quantize
                if (num_bits_do(j) == 0)
                    quant_coeff_scaled(j) = 0;
                    
                elseif (do_coeff_scaled_dyn(j) <= first_quant_point)
                    quant_coeff_scaled(j) = first_quant_point;           % Check if lies between -1 and first quantization point
                    
                elseif (do_coeff_scaled_dyn(j) >= last_quant_point)
                    quant_coeff_scaled(j) = last_quant_point;            % Check if lies between -1 and first quantization point
                    
                % Otherwise lies between first_quant_point and last_quant_point
                else
                    
                    lower_quant_idx = floor((do_coeff_scaled_dyn(j)-first_quant_point)/resolution_per_dim_do(j));     %Index of lower quantization point
                    lower_quant_point = first_quant_point + lower_quant_idx*resolution_per_dim_do(j);                     %Lower quantization point
                    upper_quant_point = first_quant_point + (lower_quant_idx+1)*resolution_per_dim_do(j);                 %Upper quantization point

                    % Do nearest neighbor quantization
                    if ((do_coeff_scaled_dyn(j) - lower_quant_point) <= resolution_per_dim_do(j)/2)
                        quant_coeff_scaled(j) = lower_quant_point;
                    else
                        quant_coeff_scaled(j) = upper_quant_point;
                    end
                    
                end   
            end
            
            quant_coeff = quant_coeff_scaled*dyn_range;
            q_do = S*quant_coeff;                          % Inverse transform from quantized coefficients
            
            % Update error (to be fed back)
            e_do = q_do - u_do;
            
            % Server takes a descent step and updates iterate
            x_hat_do = x_hat_do - eta_do*q_do;
            
        end
        
        % Final distance to optimality
        final_DistToOpt_do = norm(x_hat_do - x_star, 2);
        
        % Convergence rate
        conv_rate_DQGD_do = (final_DistToOpt_do/D)^(1/T);
        conv_rate_DQGD_do_sum = conv_rate_DQGD_do_sum + conv_rate_DQGD_do;
                
    end
    
    
    %---------------------------------------------------------------------------------------------
    % Empirical average of convergence rates over different realizations
    conv_rate_unquantized_gd_array(R_ind) = conv_rate_unquantized_gd_sum/num_realizations;
    conv_rate_DQGD_array(R_ind) = min(conv_rate_DQGD_sum/num_realizations,1);       %Clip off maximum rate at 1 
    conv_rate_DQGD_norm_separate_array(R_ind) = min(conv_rate_DQGD_norm_separate_sum/num_realizations,1);
    conv_rate_DQGD_hadamard_array(R_ind) = min(conv_rate_DQGD_hadamard_sum/num_realizations,1);
    conv_rate_DQGD_HadamardDemocratic_array(R_ind) = min(conv_rate_DQGD_HadamardDemocratic_sum/num_realizations,1);
    conv_rate_DQGD_ndo_array(R_ind) = min(conv_rate_DQGD_ndo_sum/num_realizations,1);
    conv_rate_DQGD_do_array(R_ind) = min(conv_rate_DQGD_do_sum/num_realizations,1);
    
end


%%
% Plot results
R_range = min_R:R_step:max_R;
figure;
plot(R_range, conv_rate_unquantized_gd_array, '-o');
hold on;
plot(R_range, conv_rate_DQGD_array, '-x');
plot(R_range, conv_rate_DQGD_norm_separate_array, '--');
plot(R_range, conv_rate_DQGD_hadamard_array, '-+');
plot(R_range, conv_rate_DQGD_HadamardDemocratic_array, '-*');
plot(R_range, conv_rate_DQGD_ndo_array, '-.');
plot(R_range, conv_rate_DQGD_do_array, 'color', 'k');
legend('Unquantized GD', 'Vanilla DQGD', 'DQGD Norm Separate', 'Hadamard heuristic', 'l-inf minimization', 'Near-democratic (orthonormal frame)', 'Democratic (orthonormal frame)');

%%
filename = 'Workspace_DQGD';
save(filename);