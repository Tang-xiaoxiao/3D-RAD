import os

import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pandas as pd
import json
import evaluate

model = '4b'
path_dir = f"results/{model}"

TASK = {
        'task1': ['Anatomical_observation', 'Pathological_observation'],
        'task2': ['Abnormality_feature', 'Abnormality_position', 'Abnormality_type', 'Diagnosis'],
        'task3': ['Diameter', 'Size', 'Thickness'],
        'task4': ['Arterial wall calcification', 'Atelectasis', 'Bronchiectasis', 'Cardiomegaly', 'Consolidation',
                  'Coronary artery wall calcification', 'Emphysema', 'Hiatal hernia', 'Interlobular septal thickening',
                  'Lung nodule', 'Lung opacity', 'Lymphadenopathy', 'Medical material', 'Mosaic attenuation pattern',
                  'Peribronchial thickening', 'Pericardial effusion', 'Pleural effusion', 'Pulmonary fibrotic sequela'],
        'task5': ['b', 'c', 'd', 'e', 'f', 'g', 'h'],
        'task6': ['b', 'c', 'd', 'e', 'f', 'g', 'h'],
}

def evaluate_open_subtask(task, subtask):

    print(f"-----------Evaluate {task} {subtask}------------")
    csv_path = f"{path_dir}/{task}/{subtask}/eval_open_vqa.csv"

    df = pd.read_csv(csv_path)

    preds = df['Pred'].tolist()
    answers = df['Answer'].tolist()

    preds = [str(p).strip() for p in preds]
    answers = [str(a).strip() for a in answers]

    bleu = evaluate.load('bleu')
    bleu_result = bleu.compute(predictions=preds, references=[[a] for a in answers], max_order=1)

    rouge = evaluate.load('rouge')
    rouge_result = rouge.compute(predictions=preds, references=[[a] for a in answers], rouge_types=['rouge1'])

    bertscore = evaluate.load('bertscore')
    bert_result = bertscore.compute(predictions=preds, references=[[a] for a in answers], lang="en")

    print("Overall Metrics:")
    print(f"BLEU-1: {bleu_result['bleu']:.4f}")
    print(f"ROUGE-1: {rouge_result['rouge1']:.4f}")
    print(f"BERTScore F1: {sum(bert_result['f1'])/len(bert_result['f1']):.4f}")

    results = {
        "bleu": bleu_result['bleu'],
        "rouge1": rouge_result['rouge1'],
        "f1": sum(bert_result['f1']) / len(bert_result['f1'])
    }

    result_path = f"{path_dir}/{task}/{subtask}/result.json"

    with open(result_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)

    return results


def evaluate_close_subtask(task, subtask):
    print(f"-----------Evaluate {task} {subtask}------------")
    file_path = f"{path_dir}/{task}/{subtask}/eval_close_vqa.csv"

    df = pd.read_csv(file_path)

    total = len(df)
    correct = df['Correct'].sum()
    accuracy = correct / total if total > 0 else 0

    print(f"Accuracy: {accuracy:.4f}")

    results = {
        "accuracy": accuracy,
    }

    result_path = f"{path_dir}/{task}/{subtask}/result.json"

    with open(result_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)

    return results

def evaluate_open_task(task):
    bleus, rouge1s, f1s = [], [], []
    for subtask in TASK[task]:
        result = evaluate_open_subtask(task, subtask)
        bleus.append(result["bleu"])
        rouge1s.append(result["rouge1"])
        f1s.append(result["f1"])
    bleu = np.mean(np.array(bleus))
    rouge1 = np.mean(np.array(rouge1s))
    f1 = np.mean(np.array(f1s))
    print(f"-----------Evaluate {task}------------")
    print(f"Average Metrics {task}:")
    print(f"BLEU-1: {bleu:.4f}")
    print(f"ROUGE-1: {rouge1:.4f}")
    print(f"BERTScore F1: {f1:.4f}")
    results = {
        "bleu": bleu,
        "rouge1": rouge1,
        "f1": f1
    }

    result_path = f"{path_dir}/{task}/result.json"
    with open(result_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)

def evaluate_close_task(task):
    accs = []
    for subtask in TASK[task]:
        result = evaluate_close_subtask(task, subtask)
        accs.append(result["accuracy"])
    accuracy = np.mean(np.array(accs))
    print(f"-----------Evaluate {task}------------")
    print(f"Average Metrics {task}:")
    print(f"Accuracy: {accuracy:.4f}")
    results = {"accuracy": accuracy}

    result_path = f"{path_dir}/{task}/result.json"

    with open(result_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)


def evaluate_task(task):
    if task in ['task1', 'task2', 'task3', 'task4']:
        evaluate_open_task(task)
    else:
        evaluate_close_task(task)


def evaluate_subtask(task, subtask):
    if task in ['task1', 'task2', 'task3', 'task4']:
        evaluate_open_subtask(task, subtask)
    else:
        evaluate_close_subtask(task, subtask)


if __name__ == '__main__':

    for task in TASK.keys():
        evaluate_task(task)