# [NeurIPS 2025] 🩻 3D-RAD: A Comprehensive 3D Radiology Med-VQA Dataset with Multi-Temporal Analysis and Diverse Diagnostic Tasks
<div align="center">
  <a href="https://github.com/Tang-xiaoxiao/3D-RAD/stargazers">
    <img src="https://img.shields.io/github/stars/Tang-xiaoxiao/3D-RAD?style=social" />
  </a>
  <a href="https://arxiv.org/abs/2506.11147">
    <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b.svg?logo=arxiv" />
  </a>
  <a href="https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity">
    <img src="https://img.shields.io/badge/Maintained%3F-yes-green.svg" />
  </a>
</div>

## 📢 News

<summary><strong>What's New in This Update 🚀</strong></summary>

- **2025.10.23**: 🔥 Updated **the latest version** of the paper!  
- **2025.09.19**: 🔥 Paper accepted to **NeurIPS 2025**! 🎯
- **2025.05.16**: 🔥 Set up the repository and committed the dataset!

## 🔍 Overview
💡 In this repository, we present **"3D-RAD: A Comprehensive 3D Radiology Med-VQA Dataset with Multi-Temporal Analysis and Diverse Diagnostic Tasks"**.

In our project, we collect a large-scale dataset designed to advance 3D Med-VQA using radiology CT scans, 3D-RAD, encompasses six diverse VQA tasks: **Anomaly Detection** (task 1), **Image Observation** (task 2), **Medical Computation** (task 3), **Existence Detection** (task 4), **Static Temporal Diagnosis** (task 5), and **Longitudinal Temporal Diagnosis** (task 6). 

This code can evaluate our 3D-RAD dataset on M3D and RadFM models. (OmniV model code has not been published, we will publish the evaluation code on OmniV model after they publish.) **If you find this project useful, please give it a ⭐! We encourage you to try our benchmark and welcome any feedback 💬 or collaboration 😄.**

![overview](https://github.com/Tang-xiaoxiao/3D-RAD/blob/main/Figures/overview.png)
![main](https://github.com/Tang-xiaoxiao/3D-RAD/blob/main/Figures/main.png)

## 📊 3D-RAD Dataset
In the `3DRAD` directory, there are QA data without 3D images.
You can find the full dataset with 3D images (For efficient model input, the original CT images were preprocessed and converted into .npy format.) in [3D-RAD_Dataset](https://huggingface.co/datasets/Tang-xiaoxiao/3D-RAD).

![distribution](https://github.com/Tang-xiaoxiao/3D-RAD/blob/main/Figures/distribution.png)
![construction](https://github.com/Tang-xiaoxiao/3D-RAD/blob/main/Figures/Construction.png)

## 🤖 M3D-RAD Model
To assess the utility of 3D-RAD, we **finetuned two M3D model variants** with different parameter scales, thereby constructing the M3D-RAD models. You can find our finetuned model in [M3D-RAD_Models](https://huggingface.co/Tang-xiaoxiao/M3D-RAD).

![finetuned](https://github.com/Tang-xiaoxiao/3D-RAD/blob/main/Figures/finetuned.png)

## 📈 Evaluation

### Zero-Shot Evaluation.
We conducted **zero-shot evaluation** of several stateof-the-art 3D medical vision-language models on our benchmark to assess their generalization capabilities.

![zeroshot](https://github.com/Tang-xiaoxiao/3D-RAD/blob/main/Figures/zeroshot.png)

In the `RadFM` and `M3D` directory, there are code for evaluating RadFM and M3D models on our 3D-RAD benchmark. Note that, the base code is [RadFM](https://github.com/chaoyi-wu/RadFM), and the base code is [M3D](https://github.com/BAAI-DCAI/M3D). To run our evaluation, you should first satisfy the requirements and download the models according to the base code of these models.

Compare to the base code, we make the following modifications: In the `RadFM` directory, we add a new Dataset in `RadFM/src/Dataset/dataset/rad_dataset.py` and modify the Dataset to test in `RadFM/src/Dataset/multi_dataset_test.py`. Then we add a new python file to evaluate our benchmark in `RadFM/src/eval_3DRAD.py`. In the `M3D` directory, we add a new Dataset in `M3D/Bench/dataset/multi_dataset.py` and add a new python file to evaluate our benchmark in `M3D/Bench/eval/eval_3DRAD.py`.

You can evaluate RadFM on our 3D-RAD benchmark by running:

```python
cd 3D-RAD/RadFM/src
python eval_3DRAD.py \
--file_path={your test file_path} \
--output_path={your saved output_path}
```

You can evaluate M3D on our 3D-RAD benchmark by running:

```python
cd 3D-RAD/M3D
python Bench/eval/eval_3DRAD.py \
--model_name_or_path={your model_name} \
--vqa_data_test_path={your test file_path} \
--output_dir={your saved output_dir}
```

### Scaling with Varying Training Set Sizes.
To further investigate the impact of dataset scale on model performance, we randomly **sampled 1%, 10% and 100%** of the training data per task and fine-tuned M3D accordingly. 

![varysizes](https://github.com/Tang-xiaoxiao/3D-RAD/blob/main/Figures/varysizes.png)

## 📁 Data Source
The original CT scans in our dataset are derived from [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE), which is released under a CC-BY-NC-SA license. We fully comply with the license terms by using the data for non-commercial academic research, providing proper attribution.

## 🔗 Model Links

| Model | Paper                                                        |
| ----- | ------------------------------------------------------------ |
| [RadFM](https://github.com/chaoyi-wu/RadFM) | Towards Generalist Foundation Model for Radiology by Leveraging Web-scale 2D&3D Medical Data | https://github.com/chaoyi-wu/RadFM |
| [M3D](https://github.com/BAAI-DCAI/M3D)   | M3D: Advancing 3D Medical Image Analysis with Multi-Modal Large Language Models |
| OmniV(not open) | OmniV-Med: Scaling Medical Vision-Language Model for Universal Visual Understanding |
