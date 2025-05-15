import csv
import json
import logging
import os
import re
import difflib
import sys
import cv2
import torch
import random
from abc import abstractmethod
from itertools import islice
from scipy import ndimage
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from collections.abc import Mapping
from torch.utils.data import DataLoader
import PIL
import SimpleITK as sitk
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from collections import defaultdict
from PIL import Image
import math


class RAD_Dataset(Dataset):
    """_summary_

    Args:
        Dataset (_type_): _description_: caption task formulated as vqa task for Radiopaedia dataset
        csv_path (_type_): path to csv file
        prompt_json_file (_type_): path to json file containing caption prompts
    Output:
        Dict: {
             "image_dict": {"image": image, "position": {"question": 0}}, # image is a tensor of shape [s,c,w,h,d] like, [1,3,512,512,1], position is a dict, random choice of 0 or len(question)
            "question": question, # random choice of caption prompts
            "answer":answer, # caption
            }
    """
    def __init__(self, csv_path):
        data_info = pd.read_csv(csv_path)
        # npy_path,image_caption,question,answer
        self.data_list = data_info
        self.data_root_df = pd.read_csv("../../valid_path.csv")
        self.data_root_df.set_index('VolumeName', inplace=True)
        # self.img_path_list = np.asarray(data_info['image_path'])
        # self.question_list = np.asarray(data_info['question'])
        # self.answer_list = np.asarray(data_info['answer'])
    
    def __len__(self):
        # return len(self.img_path_list)
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list.iloc[index]
        volume_name = data['VolumeName']
        img_path = self.data_root_df.at[volume_name, 'Path']
        image = np.load(img_path)

        image = (image-image.min())/(image.max()-image.min())
        contain_nan = (True in np.isnan(image))
        if contain_nan:
            image = np.random.randn(3,512,512,4)

        image = torch.from_numpy(image).float()
        question = data["Question"]

        # 判断是否有选项
        if all(x in data for x in ["Choice A", "Choice B", "Choice C", "Choice D"]):
            choices = "Choices: A. {} B. {} C. {} D. {}".format(
                data["Choice A"], data["Choice B"], data["Choice C"], data["Choice D"]
            )
            answer = "{}. {}".format(data["AnswerChoice"], data["Answer"])
        elif all(x in data for x in ["Choice A", "Choice B"]):
            choices = "Choices: A. {} B. {}".format(data["Choice A"], data["Choice B"])
            answer = "{}. {}".format(data["AnswerChoice"], data["Answer"])
        else:
            choices = ""
            answer = str(data["Answer"])

        # 拼接问题
        if choices:
            question = question + '\n' + choices + '\n' + 'Answer:'
        else:
            question = question + '\nAnswer:'

        image_dict = {
            "image": image,
            "position": {
                "question": 0
            }
        }
        # if random.random() < 0.5:
        #     image_dict = {
        #         "image": image,
        #         "position": {
        #             "question": 0
        #         }
        #     }
        # else:
        #     image_dict = {
        #         "image": image,
        #         "position": {
        #             "question": len(question)
        #         }
        #     }

        return {
            "image_dict": [image_dict],
            "question": question,
            "answer": answer,
            }
