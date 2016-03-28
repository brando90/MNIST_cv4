%gau_precision = 0.5
num_inits = 5
iterations = int64(1000)
% train_func_name = 'learn_HBF1_SGD'
% mdl_func_name = 'HBF1'
%train_func_name = 'learn_RBF_SGD'
%mdl_func_name = 'RBF'
train_func_name = 'learn_HReLu_SGD'
mdl_func_name = 'HReLu'
lambda = 0
eta_c = 0.01
eta_t = 0.01
visualize = 0
sgd_errors = 1
%% locations
cp_folder = 'cp_28mar_omj_HReLu1/'
cp_param_files_names = 'cp_28mar_omj_HReLu1_%d.m'
results_path = './results/r_28mar_omj_HReLu1/'
%% jobs
jobs = 5
start_centers = 10
end_centers = 250
%% data
data_set_path = '../../hbf_research_data/data_MNIST_0.7_0.15_0.15_49000_10500_10500.mat'
data_normalized = 0
%% GPU
gpu_on = 1
%% inits
rbf_as_initilization = 0
c_init_normalized = 0
