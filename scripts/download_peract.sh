# ALL TASKSK
# close_jar
# insert_onto_square_peg
# light_bulb_in
# meat_off_grill
# open_drawer
# place_cups
# place_shape_in_shape_sorter
# place_wine_at_rack_location
# push_buttons
# put_groceries_in_cupboard
# put_item_in_drawer
# put_money_in_safe
# reach_and_drag
# slide_block_to_color_target
# stack_blocks
# stack_cups
# sweep_to_dustpan_of_size
# turn_tap


tasks=open_drawer,put_item_in_drawer,stack_blocks,turn_tab,sweep_to_dustpan_of_size,reach_and_drag,close_jar

save_path=/home/felix/Workspace/diffusion_policy_felix/data/peract

mkdir -p $save_path
mkdir -p $save_path/train
mkdir -p $save_path/val
mkdir -p $save_path/test

echo 'Starting download of training tasks'
for task in $(echo $tasks | tr ',' '\n'); do
    echo 'Downloading task: ' $task
    rclone copy gdrive,shared_with_me:rlbench/train/${task}.zip ${save_path}/train/
    unzip ${save_path}/train/${task}.zip -d ${save_path}/train/
    rm ${save_path}/train/${task}.zip
done

echo 'Starting download of validation tasks'
for task in $(echo $tasks | tr ',' '\n'); do
    echo 'Downloading task: ' $task
    rclone copy gdrive,shared_with_me:rlbench/val/${task}.zip ${save_path}/val/
    unzip ${save_path}/val/${task}.zip -d ${save_path}/val/
    rm ${save_path}/val/${task}.zip
done

echo 'Starting download of test tasks'
for task in $(echo $tasks | tr ',' '\n'); do
    echo 'Downloading task: ' $task
    rclone copy gdrive,shared_with_me:rlbench/test/${task}.zip ${save_path}/test/
    unzip ${save_path}/test/${task}.zip -d ${save_path}/test/
    rm ${save_path}/test/${task}.zip
done
