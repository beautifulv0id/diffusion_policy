#!/bin/bash

training_script=$1
training_script_base="${training_script:0:${#training_script}-3}"
task_config=$2

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
path="data/outputs/${year}.${month}.${day}/${hour}.${minute}.${second}_${training_script_base}_${task_config}"

# Print the path
echo "$path"
