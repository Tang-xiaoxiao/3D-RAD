# M3D-RAD
The official code for the paper "3D-RAD: A Comprehensive 3D Radiology Med-VQA Dataset with Multi-Temporal Analysis and Diverse Diagnostic Tasks".

In our project, we collect a large-scale dataset designed to advance 3D Med-VQA using radiology CT scans, 3D-RAD, encompasses six diverse VQA tasks: anomaly detection (task 1), image observation (task 2), medical computation (task 3), existence detection (task 4), static temporal diagnosis (task 5), and longitudinal temporal diagnosis (task 6). 

This code can evaluate our 3D-RAD dataset on M3D and RadFM models. (OmniV model code has not been published, we will publish the evaluation code on OmniV model after they publish.)

![main](https://github.com/Tang-xiaoxiao/M3D-RAD/blob/main/Figures/main.png)

## 3D-RAD Dataset
In the `3DRAD` directory, there are QA data without 3D images.
You can find the dataset in https://huggingface.co/datasets/Tang-xiaoxiao/3D-RAD

## Evaluation
In the `RadFM` and `M3D` directory, there are code for evaluating RadFM and M3D models on our 3D-RAD benchmark. Note that, the base code in RadFM is  from https://github.com/chaoyi-wu/RadFM, and the base code in M3D is from https://github.com/BAAI-DCAI/M3D. To run our evaluation, you should first satisfy the requirements and download the models according to the base code of these models.

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

### Model Links

| Model | Paper                                                        | Link                               |
| ----- | ------------------------------------------------------------ | ---------------------------------- |
| RadFM | Towards Generalist Foundation Model for Radiology by Leveraging Web-scale 2D&3D Medical Data | https://github.com/chaoyi-wu/RadFM |
| M3D   | M3D: Advancing 3D Medical Image Analysis with Multi-Modal Large Language Models | https://github.com/BAAI-DCAI/M3D   |
| OmniV | OmniV-Med: Scaling Medical Vision-Language Model for Universal Visual Understanding | Has not been published             |
