#!/bin/bash

training_script=$1
training_script="${training_script:0:${#training_script}-3}"

task_name=$2

# Get current date and time
datetime=$(date '+%Y%m%d%H%M%S')

# Extract components
year=${datetime:0:4}
month=${datetime:4:2}
day=${datetime:6:2}
hour=${datetime:8:2}
minute=${datetime:10:2}
second=${datetime:12:2}

# Construct the path
path="${DIFFUSION_POLICY_ROOT}/data/outputs/${year}.${month}.${day}/${hour}.${minute}.${second}_${training_script}_${task_name}"

# Print the path
echo "$path"
