import os

general_filepaths = ['/media/funk/INTENSO/300_RL_BENCH/train_13_01/prev_trains', '/media/funk/INTENSO/300_RL_BENCH/train_13_01', '/media/funk/INTENSO/300_RL_BENCH/train_13_01/trains_14_01']
folders = []
for i in range(len(general_filepaths)):
    top_lvl_folder = general_filepaths[i] + '/'
    # get all the folders:
    if (os.path.isdir(top_lvl_folder)):
        folders_tmp = os.listdir(top_lvl_folder)
        # convert to list:
        folders.extend([top_lvl_folder + folder for folder in folders_tmp])

# now do a selection:
# selected_folders = ['21.06.30_train_diffuser_actor_open_drawer_image', '21.07.27_train_se3_flow_matching_fms1_open_drawer_image', '21.09.10_train_se3_flow_matching_fms1_standard_trafo_open_drawer_image']
# plot_names = ['3DDA', 'URSA', 'NormalTrafo']

# # now do a selection:
# selected_folders = ['21.06.30_train_diffuser_actor_open_drawer_image', '21.07.27_train_se3_flow_matching_fms1_open_drawer_image',\
#                     '00.03.52_train_se3_flow_matching_fms1_open_drawer_image_normal', '22.14.02_train_se3_flow_matching_fms1_open_drawer_image_npoints250', '23.16.04_train_se3_flow_matching_fms1_open_drawer_image_npoints500',\
#                     '22.05.02_train_se3_flow_matching_fms1_open_drawer_image_npoints2000', '23.20.20_train_se3_flow_matching_fms1_open_drawer_image_cfg1', '23.22.42_train_se3_flow_matching_fms1_open_drawer_image_cfg2',\
#                     '21.09.10_train_se3_flow_matching_fms1_standard_trafo_open_drawer_image']
# plot_names = ['3DDA', 'URSA', 'URSA Normal Vec', 'URSA 250',  'URSA 500', 'URSA 2000', 'URSA cfg1', 'URSA cfg2', 'NormalTrafo']

# now do a selection:
selected_folders = ['21.06.30_train_diffuser_actor_open_drawer_image', \
                    '21.07.27_train_se3_flow_matching_fms1_open_drawer_image',\
                    '00.03.52_train_se3_flow_matching_fms1_open_drawer_image_normal', \
                    # '22.14.02_train_se3_flow_matching_fms1_open_drawer_image_npoints250', \
                    # '23.16.04_train_se3_flow_matching_fms1_open_drawer_image_npoints500',\
                    # '22.05.02_train_se3_flow_matching_fms1_open_drawer_image_npoints2000', \
                    # '23.20.20_train_se3_flow_matching_fms1_open_drawer_image_cfg1', \
                    '23.22.42_train_se3_flow_matching_fms1_open_drawer_image_cfg2',\
                    '10.07.43_train_se3_flow_matching_fms1_open_drawer_image_cfg2_2000',\
                    '10.42.40_train_se3_flow_matching_fms1_open_drawer_image_cfg2_3000',\
                    '21.09.10_train_se3_flow_matching_fms1_standard_trafo_open_drawer_image']
plot_names = ['3DDA', 'URSA', 'URSA Normal Vec', 'URSA cfg2_1000', 'URSA cfg2_2000', 'URSA cfg2_3000',  'NormalTrafo']



folders_with_eval = []
eval_file = []
# now go though all the folders and check whether there is an evaluation folder in them
for folder in folders:
    if (os.path.isdir(folder)):
        curr_folders = os.listdir(folder)
        for curr_folder_i in curr_folders:
            if ('eval' in curr_folder_i):
                # check if a specific eval file exists
                if (os.path.exists(folder + '/' + curr_folder_i + '/' + 'eval_logs.json.txt')):
                    folders_with_eval.append(folder)
                    eval_file.append(folder + '/' + curr_folder_i + '/' + 'eval_logs.json.txt')

# now go through all the eval files and check whether the training was successful
print ("WATI")
train_metrics_list = []
eval_metrics_list = []
epoch_list = []
for eval_file_i in eval_file:
    tmp_train_metrics_list = []
    tmp_eval_metrics_list = []
    tmp_epoch_list = []
    with open(eval_file_i, 'r') as f:
        lines = f.readlines()
        for line in lines:
            res_str = line.split(',')
            tmp_train_metrics_list.append(float(res_str[0].split(':')[1]))
            tmp_eval_metrics_list.append(float(res_str[1].split(':')[1]))
            tmp_epoch_list.append(int(res_str[2].split(':')[-1][:-2]))
        train_metrics_list.append(tmp_train_metrics_list)
        eval_metrics_list.append(tmp_eval_metrics_list)
        epoch_list.append(tmp_epoch_list)

folders_with_eval_tmp = []
train_metrics_list_tmp = []
eval_metrics_list_tmp = []
epoch_list_tmp = []
run_name_tmp = []

for i in range(len(folders_with_eval)):
    for j in range(len(selected_folders)):
        if (selected_folders[j] in folders_with_eval[i]):
            folders_with_eval_tmp.append(folders_with_eval[i])
            train_metrics_list_tmp.append(train_metrics_list[i])
            eval_metrics_list_tmp.append(eval_metrics_list[i])
            epoch_list_tmp.append(epoch_list[i])
            run_name_tmp.append(plot_names[j])

# now plot the stuff:
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

print ("Train Performance")
for i in range(len(folders_with_eval_tmp)):
    print (run_name_tmp[i])
    print ("Mean: " + str(np.mean(train_metrics_list_tmp[i])))
    print ("Std: " + str(np.std(train_metrics_list_tmp[i])))

print ("Test Performance")
for i in range(len(folders_with_eval_tmp)):
    print (run_name_tmp[i])
    print ("Mean: " + str(np.mean(eval_metrics_list_tmp[i])))
    print ("Std: " + str(np.std(eval_metrics_list_tmp[i])))

mean_train_success = []
std_train_success = []
mean_eval_success = []
std_eval_success = []
for i in range(len(folders_with_eval_tmp)):
    mean_train_success.append(np.mean(train_metrics_list_tmp[i]))
    std_train_success.append(np.std(train_metrics_list_tmp[i]))
    mean_eval_success.append(np.mean(eval_metrics_list_tmp[i]))
    std_eval_success.append(np.std(eval_metrics_list_tmp[i]))

# create a bar plot for train and eval metrics
barWidth = 0.3
r1 = np.arange(len(folders_with_eval_tmp))
r2 = [x + barWidth for x in r1]

plt.bar(r1, mean_train_success, color='tab:gray', width=barWidth, edgecolor='grey', label='train')
# add error in bar
plt.errorbar(r1, mean_train_success, yerr=std_train_success, fmt='o', color='black', elinewidth=2, capsize=5, capthick=2)
plt.bar(r2, mean_eval_success, color='b', width=barWidth, edgecolor='grey', label='eval')
plt.errorbar(r2, mean_eval_success, yerr=std_eval_success, fmt='o', color='black', elinewidth=2, capsize=5, capthick=2)


plt.xlabel('runs', fontweight='bold')
plt.ylabel('success rate', fontweight='bold')
plt.xticks([r + barWidth-0.5*barWidth for r in range(len(folders_with_eval_tmp))], run_name_tmp)
plt.legend()
plt.show()