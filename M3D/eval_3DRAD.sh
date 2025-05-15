#!/bin/bash
export CUDA_VISIBLE_DEVICES=6
# Define model
model='7b'
model_name_or_path='GoodBaiBai88/M3D-LaMed-Llama-2-7B'
TASK_todo=("task1" "task2" "task3" "task4" "task5" "task6" "task7")

# Define the tasks and subtasks

task1=("Anatomical_observation" "Pathological_observation")
task2=("Abnormality_feature" "Abnormality_position" "Abnormality_type" "Diagnosis")
task3=("Diameter" "Size" "Thickness")
task4=("Arterial wall calcification" "Atelectasis" "Bronchiectasis" "Cardiomegaly" "Consolidation" \
               "Coronary artery wall calcification" "Emphysema" "Hiatal hernia" "Interlobular septal thickening" \
               "Lung nodule" "Lung opacity" "Lymphadenopathy" "Medical material" "Mosaic attenuation pattern" \
               "Peribronchial thickening" "Pericardial effusion" "Pleural effusion" "Pulmonary fibrotic sequela")
task5=("b" "c" "d" "e" "f" "g" "h")
task6=("b" "c" "d" "e" "f" "g" "h")

# Loop through each task and its subtasks
for task in "${TASK_todo[@]}"; do
    # Set task_list based on the current task
    if [[ "$task" == "task1" ]]; then
      task_list=("${task1[@]}")
    elif [[ "$task" == "task2" ]]; then
      task_list=("${task2[@]}")
    elif [[ "$task" == "task3" ]]; then
      task_list=("${task3[@]}")
    elif [[ "$task" == "task4" ]]; then
      task_list=("${task4[@]}")
    elif [[ "$task" == "task5" ]]; then
      task_list=("${task5[@]}")
    elif [[ "$task" == "task6" ]]; then
      task_list=("${task6[@]}")
    else
      task_list=("${task7[@]}")
    fi

    # Loop through each subtask for the current task
    for subtask in "${task_list[@]}"; do
        echo "--------------Evaluate ${task} ${subtask}------------"

        # Define paths
        vqa_data_test_path="../3DRAD/test/${task}/${subtask}.csv"
        output_dir="../results/${model}/${task}/${subtask}/"

        # Set close_ended flag based on the task
        if [[ "$task" == "task1" || "$task" == "task2" || "$task" == "task3" || "$task" == "task4" ]]; then
            python eval_vqa_for_bash.py --vqa_data_test_path "$vqa_data_test_path" --output_dir "$output_dir" --model_name_or_path "$model_name_or_path"
        else
            python eval_vqa_for_bash.py --vqa_data_test_path "$vqa_data_test_path" --output_dir "$output_dir" --model_name_or_path "$model_name_or_path" --close_ended
        fi

    done
done
