clear;
task_id = 8
%%
config_name = 'simulation_config_pca.m'
run(config_name);
run('load_paths.m');
%% load configs
current_simulation_config = sprintf( './changing_params/%s%s', cp_folder, config_name )
run(current_simulation_config);
%% load data set
load(data_set_path); % data4cv
if data_normalized
    disp('DATA NORMALIZED')
    data4cv.normalize_data();
end
[ X_train,X_cv,X_test, y_train,y_cv,y_test ] = data4cv.get_data_for_hold_out_cross_validation();
%% get models
results_path = results_path(2:length(results_path)-1 )
mdl_location = sprintf('.%s/test_error_vs_iterations%d',results_path,task_id);
load(mdl_location)
%%
%location = sprintf('train_mnist_reconstruction_task_id_%d', task_id);
%mkdir( sprintf('./%s',location) )
num_test_reconstruction = 5
for img_num_train=1:num_test_reconstruction
    mnist_image_train = X_test(:,img_num_train);
    reconstruction_PCA = U * U' * mnist_image_train ;
    %% original 
    original_fig = figure;
    display_network( mnist_image_train );
    title('Original train Image')
    %% HBF1 reconstruct
    pca_fig = figure;
    display_network( reconstruction_PCA );
    title(sprintf('train reconstruction%d', task_id))
    colorbar
    title(sprintf('PCA train treconstruction task id = %d',task_id))
    %% save
%      saveas(original_fig, sprintf( './%s/img_num_%d_a_original.jpeg',location, img_num_train) );
%      saveas(original_fig, sprintf('./%s/img_num_%d_a_original',location, img_num_train) );
%      saveas(hbf1_fig, sprintf( './%s/img_num_%d_pca_center_%d.jpeg',location, img_num_train, center) );
%      saveas(hbf1_fig, sprintf('./%s/img_num_%d_pca_center_%d',location, img_num_train, center) );
%     saveas(rbf_fig, sprintf( './%s/img_num_%d_rbf_center_%d.jpeg',location, img_num_train, center) );
%     saveas(rbf_fig, sprintf('./%s/img_num_%d_rbf_center_%d',location, img_num_train, center) );
end
% %%
% location = sprintf('test_mnist_reconstruction_task_id_%d', task_id);
% mkdir( sprintf('./%s',location) )
% num_test_reconstruction = 15
% for img_num_train=1:num_test_reconstruction
%     mnist_image_train = X_test(:,img_num_train);
%     reconstruction_HBF1 = best_iteration_mdl.predict( mnist_image_train );
%     reconstruction_rbf = rbf_mdl.predict( mnist_image_train );
%     %% original 
%     original_fig = figure;
%     display_network( mnist_image_train );
%     title('Original test Image')
%     %% HBF1 reconstruct
%     hbf1_fig = figure;
%     display_network( reconstruction_HBF1 );
%     title(sprintf('test reconstruction%d', task_id))
%     colorbar
%     title(sprintf('HBF1 test reconstruction task id = %d',task_id))
%     %%
%     rbf_fig = figure;
%     display_network( reconstruction_rbf );
%     title(sprintf('test reconstruction%d', task_id))
%     title(sprintf('RBF test reconstruction task id = %d',task_id))
%     colorbar
%     saveas(original_fig, sprintf( './%s/img_num_%d_a_original.jpeg',location, img_num_train) );
%     saveas(original_fig, sprintf('./%s/img_num_%d_a_original',location, img_num_train) );
%     saveas(hbf1_fig, sprintf( './%s/img_num_%d_hbf_center_%d.jpeg',location, img_num_train, center) );
%     saveas(hbf1_fig, sprintf('./%s/img_num_%d_hbf_center_%d',location, img_num_train, center) );
%     saveas(rbf_fig, sprintf( './%s/img_num_%d_rbf_center_%d.jpeg',location, img_num_train, center) );
%     saveas(rbf_fig, sprintf('./%s/img_num_%d_rbf_center_%d',location, img_num_train, center) );
% end