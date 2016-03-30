function [] = get_best_trained_1layered_model(slurm_job_id, task_id)
restoredefaultpath
slurm_job_id
task_id
%% load the path to configs for this simulation
run('./simulation_config.m');
run('load_paths.m');
%% load most of the configs for this model
current_simulation_config = sprintf( './changing_params/%s%s', cp_folder, 'simulation_config.m' )
run(current_simulation_config);
%% load the number of centers for this model
changing_params_for_current_task = sprintf( sprintf('./changing_params/%s%s',cp_folder,cp_param_files_names), task_id )
run(changing_params_for_current_task);
%% load data set
load(data_set_path); % data4cv
if data_normalized
    data4cv.normalize_data();
end
[ X_train,X_cv,X_test, y_train,y_cv,y_test ] = data4cv.get_data_for_hold_out_cross_validation();
%% preparing models to train/test for mdl_iterator
center
%% rand seed
rand_seed = get_rand_seed( slurm_job_id, task_id)
rng(rand_seed); %rand_gen.Seed
%% Run Hold Out Cross Validation
tic;
[D, N] = size(X_train);
[D_out, ~] = size(y_test);
if strcmp( train_func_name, 'learn_HSig_SGD') || strcmp( train_func_name, 'learn_ReLu_SGD')
    [~, N_train] = size(X_train);
    [~, N_cv] = size(X_cv);
    [~, N_test] = size(X_test);
    X_train = [X_train; ones(1,N_train)];
    X_cv = [X_cv; ones(1,N_cv)];
    X_test = [X_test; ones(1,N_test)];
    D = D+1;
end
if gpu_on
    X_train = gpuArray(X_train);
    y_train = gpuArray(y_train);
    X_cv = gpuArray(X_cv);
    y_cv = gpuArray(y_cv);
    X_test = gpuArray(X_test);
    y_test = gpuArray(y_test);
end
K = center;
%% statistics for initilizations preparation
y_std = std(y_train,0,2); % (D_out x 1) unbiased std of coordinate/var/feature
y_mean = mean(y_train,2); % (D_out x 1) mean of coordinate/var/feature
y_std = repmat( y_std', [K,1]); % (K x D_out) for c = (K x D_out)
y_mean = repmat( y_mean', [K,1]); % (K x D_out) for c = (K x D_out)  
%min_y = min(y_train);
%max_y = max(y_train, 2);
%% tracking best mdl
error_best_mdl_on_cv = inf;
best_H_mdl = -1;
for initialization_index=1:num_inits
    fprintf('initialization_index: %s\n', initialization_index)
    %% t_initilization
    fprintf('t_initilization: %s\n', t_initilization);
    switch t_initilization
    case 't_random_data_points'
        t_init = datasample(X_train', K, 'Replace', false)'; % (D x K)
    case 't_zeros_plus_eps'
        t_init = normrnd(0,epsilon_t,[D,K]); % (D x K)
    case 't_random_data_points_treat_offset_special'
        %if we used normal data points all the offsets would be 1 since x=[x,1]. To avoid that we treat offset as a bit less than average inner product   
        t_init = datasample(X_train', K, 'Replace', false)'; % (D x K)
        t_mean  = norm(mean(t_init(1:D-1,:) ,2),2)^2;
        t_std = norm(std(t_init(1:D-1,:),0,2),2)^2;
        t_init(D,:) =  - normrnd(epsilon_t*t_mean - epsilon_t*t_std, t_std); % (1 x K)
    otherwise
        disp('OTHERWISE');
        error('The t wieghts init that you gace is invalid var t_initilization: %s', t_initilization);
    end
    %% c_initilization
    fprintf('c_initilization: %s\n', c_initilization);
    c_init = (1 + 1)*rand(K,D_out) - 1;
    if gpu_on
        c_init = gpuArray(c_init);
        t_init = gpuArray(t_init);
    end
    switch c_initilization
    case 'c_kernel_mdl_as_initilization'
        switch train_func_name
            case 'learn_HBF1_SGD'
                kernel_mdl = RBF(c_init,t_init,gau_precision, best_H_mdl.lambda);
                kernel_mdl = learn_RBF_linear_algebra( X_train, y_train, kernel_mdl);
                c_init = kernel_mdl.c;
            case 'learn_RBF_SGD'
                kernel_mdl = RBF(c_init,t_init,gau_precision, best_H_mdl.lambda);
                kernel_mdl = learn_RBF_linear_algebra( X_train, y_train, kernel_mdl);
                c_init = kernel_mdl.c;
            case 'learn_HSig_SGD'
                kernel_mdl = HSig(c_init,t_init,lambda);
                similarity_matrix = X_train' * kernel_mdl.t; % (N x K)
                Kern_matrix = sigmf(similarity_matrix, [-1, 0]); % (N x K)
                C = Kern_matrix \ y_train';  % (K x D) = (N x K)' x (N x D)
                kernel_mdl.c = C; % (K x D)
                c_init = kernel_mdl.c;
            case 'learn_HReLu_SGD'
                kernel_mdl = HReLu(c_init,t_init,lambda);
                similarity_matrix = X_train' * kernel_mdl.t; % (N x K)
                neuron_activated = (similarity_matrix >= 0); % (N x K)
                Kern_matrix = similarity_matrix .* neuron_activated;
                C = Kern_matrix \ y_train';  % (K x D) = (N x K)' x (N x D)
                kernel_mdl.c = C; % (K x D)
                c_init = kernel_mdl.c;
            otherwise
                disp('OTHERWISE');
                error('The train function you gave: %s does not exist', train_func_name);
         end
    case 'c_normal_zeros_plus_eps'
        c_init = normrnd(0,epsilon_c,[K,D_out]); % (K x D_out)
    case 'c_normal_ymean_ystd'
        c_init = normrnd(y_mean,y_std,[K,D_out]); % (K x D_out)
    case 'c_uniform_random_centered_ymean_std_ystd'
        c_init = (y_std + y_std) .* rand(K,D_out) + y_mean;
    case 'c_uniform_random_centered_ymean_std_min_max_y'
        %c_init = repmat(max_y - min_y,[K,1]) .* rand(K,D_out) + repmat(y_mean,[K,1]);
        error('TODO');
    case 'c_hard_coded_c_init'
        c_init = (1 + 1)*rand(K,D_out) - 1;
    otherwise
        disp('OTHERWISE')
    end
    if c_init_normalized && ~strcmp( train_func_name, 'c_kernel_mdl_as_initilization')
        %doesn't make sense to normalize the kernel c weights
        c_init = normc(c_init);
    end
    %% train mdl
    switch train_func_name
    case 'learn_HBF1_SGD'
        mdl = HBF1(c_init,t_init,gau_precision,lambda);
        [ mdl, errors_train, errors_test ] = learn_HBF1_SGD( X_train, y_train, mdl, iterations,visualize, X_test,y_test, eta_c,eta_t, sgd_errors);
    case 'learn_RBF_SGD'
        mdl = RBF(c_init,t_init,gau_precision,lambda);
        [ mdl, errors_train, errors_test ] = learn_RBF_SGD( X_train, y_train, mdl, iterations,visualize, X_test,y_test, eta_c, sgd_errors);
    case 'learn_HSig_SGD'
        mdl = HSig(c_init,t_init,lambda);
        [ mdl, errors_train, errors_test ] = learn_HSig_SGD( X_train, y_train, mdl, iterations,visualize, X_test,y_test, eta_c, eta_t, sgd_errors);
    case 'learn_HReLu_SGD'
        mdl = HReLu(c_init,t_init,lambda);
        [ mdl, errors_train, errors_test ] = learn_HReLu_SGD( X_train, y_train, mdl, iterations,visualize, X_test,y_test, eta_c, eta_t, sgd_errors);
    otherwise
       error('The train function you gave: %s does not exist', train_func_name);
    end
    %% update best mdl
    error_mdl_new_on_cv = compute_Hf_sq_error(X_cv,y_cv, mdl, mdl.lambda );
    if error_mdl_new_on_cv < error_best_mdl_on_cv
        best_H_mdl = mdl;
        error_best_mdl_on_cv = error_mdl_new_on_cv;
        best_train_error_H_mdl = errors_train;
        best_test_error_H_mdl = errors_test;
        %c_best = c_init;
        %t_best = t_init;
    end
end
%% Kernel model train
t_init = best_H_mdl.t;
c_init = best_H_mdl.c;
if gpu_on
    c_init = gpuArray(c_init);
    t_init = gpuArray(t_init);
end
switch train_func_name % get KERNEL MODEL
    case 'learn_HBF1_SGD'
        kernel_mdl = RBF(c_init,t_init,gau_precision, best_H_mdl.lambda);
        kernel_mdl = learn_RBF_linear_algebra( X_train, y_train, kernel_mdl);
    case 'learn_RBF_SGD'
        kernel_mdl = RBF(c_init,t_init,gau_precision, best_H_mdl.lambda);
        kernel_mdl = learn_RBF_linear_algebra( X_train, y_train, kernel_mdl);
    case 'learn_HSig_SGD'
        kernel_mdl = HSig(c_init,t_init,lambda);
        similarity_matrix = X_train' * kernel_mdl.t; % (N x K)
        Kern_matrix = sigmf(similarity_matrix, [-1, 0]); % (N x K)
        C = Kern_matrix \ y_train';  % (K x D) = (N x K)' x (N x D)
        kernel_mdl.c = C; % (K x D)
    case 'learn_HReLu_SGD'
        kernel_mdl = HReLu(c_init,t_init,lambda);
        similarity_matrix = X_train' * kernel_mdl.t; % (N x K)
        neuron_activated = (similarity_matrix >= 0); % (N x K)
        Kern_matrix = similarity_matrix .* neuron_activated;
        C = Kern_matrix \ y_train';  % (K x D) = (N x K)' x (N x D)
        kernel_mdl.c = C; % (K x D)
    otherwise
        disp('OTHERWISE');
        error('The train function you gave: %s does not exist', train_func_name);
 end
%% Errors of models
test_error_kernel_mdl = compute_Hf_sq_error(X_test,y_test, kernel_mdl, kernel_mdl.lambda )
train_error_kernel_mdl = compute_Hf_sq_error(X_train,y_train, kernel_mdl, kernel_mdl.lambda )
train_error_H_mdl = compute_Hf_sq_error(X_train,y_train, best_H_mdl, best_H_mdl.lambda )
test_error_H_mdl = compute_Hf_sq_error(X_test,y_test, best_H_mdl, best_H_mdl.lambda )
best_H_mdl = best_H_mdl.gather();
kernel_mdl = kernel_mdl.gather();
%% save everything/write errors during iterations
[s,git_hash_string_mnist_cv4] = system('git -C . rev-parse HEAD')
[s,git_hash_string_hbf_research_data] = system('git -C ../../hbf_research_data rev-parse HEAD')
[s,git_hash_string_hbf_research_ml_model_library] = system('git -C ../../hbf_research_ml_model_library rev-parse HEAD')
vname=@(x) inputname(1);
error_iterations_file_name = sprintf('test_error_vs_iterations%d',task_id);
path_error_iterations = sprintf('%s%s',results_path,error_iterations_file_name)
save(path_error_iterations, vname(best_train_error_H_mdl),vname(best_test_error_H_mdl), vname(center), vname(iterations), vname(eta_c), vname(eta_t), vname(best_H_mdl), vname(kernel_mdl), vname(rand_seed), vname(git_hash_string_mnist_cv4), vname(git_hash_string_hbf_research_data), vname(git_hash_string_hbf_research_ml_model_library) );
%% write results to file
result_file_name = sprintf('results_om_id%d.m',task_id);
results_path
result_path_file = sprintf('%s%s',results_path,result_file_name)
[fileID,~] = fopen(result_path_file, 'w')
fprintf(fileID, 'task_id=%d;\ncenter=%d;\ntest_error_H_mdl=%d;\ntrain_error_H_mdl=%d;\ntest_error_kernel_mdl=%d;\ntrain_error_kernel_mdl=%d;\n', task_id,center,test_error_H_mdl,train_error_H_mdl,test_error_kernel_mdl,train_error_kernel_mdl);
time_passed = toc;
%% save my own code
my_self = 'get_best_trained_1layered_model.m';
source = sprintf('./%s', my_self);
destination = sprintf('%s', results_path)
copyfile(source, destination);
%% write time elapsed to file
[secs, minutes, hours, ~] = time_elapsed(iterations, time_passed )
time_file_name = sprintf('time_duration_om_id%d.m',task_id);
path_file = sprintf('%s%s',results_path,time_file_name);
fileID = fopen(path_file, 'w')
fprintf(fileID, 'task_id=%d;\nsecs=%d;\nminutes=%d;\nhours=%d;\niterations=%d;\ncenter=%d;\ndata_set= ''%s'' ;', task_id,secs,minutes,hours,iterations,center,data_set_path);
disp('DONE');
disp('DONE training model')
end