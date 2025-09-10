from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
from PIL import Image
import random
import json

from .brain_vq_dataset import VQ_TOKENIZER_LEN, VQ_VQ_LEN, VQ_CODE_BOOK_SIZE, bcolors

from pathlib import Path
import _pickle as cPickle


class BrainVqaDataset(Dataset):       
    def __init__(self, tokenizer_len: int, mode: str):
        assert tokenizer_len == VQ_TOKENIZER_LEN

        self.mode = mode
        
        with open("total_VQA.json", "r") as f:
            brain_info = json.load(f)

        with open("total_f16.pkl", "rb") as f:
            self.brain_toks  = cPickle.load(f)

        self.outputs = []
        for i in range(len(brain_info)):
            row = brain_info[i]
            conversation_pairs = self.parse_conversations(row["conversations"])
            n_sample = 1
            sampled_pairs = random.sample(conversation_pairs, k=min(len(conversation_pairs), n_sample))

            image_path = row["path"]
            brain_vq = self.brain_toks[image_path]
            brain_vq_shifted = [x + tokenizer_len for x in brain_vq]

            for pair in sampled_pairs:
                response = pair["response"]
                instruction = pair["instruction"]

                if type(response) == float:
                    print(f"{bcolors.FAIL}Row {i} has a float report{bcolors.ENDC} {image_path}")
                    continue

                if type(instruction) == float:
                    print(f"{bcolors.FAIL}Row {i} has a float instruction{bcolors.ENDC} {image_path}")
                    continue

                self.outputs.append({"report": response, 
                                    "brain_vq_shifted": brain_vq_shifted,
                                    "brain_vq_output": None,
                                    "brain_vq_seg": None,
                                    "io_type": "input",
                                    "instruction": instruction})

    def parse_conversations(self, conversations):
        qa_pairs = []
        i = 0
        while i < len(conversations) - 1:
            current_msg = conversations[i]
            next_msg = conversations[i+1]

            if current_msg["from"] == "human" and next_msg["from"] == "gpt":
                question = current_msg["value"]
                answer = next_msg["value"]

                qa_pairs.append({
                    "instruction": question,
                    "response": answer,
                })
                i += 2
            else:
                i += 1

        return qa_pairs

    def __len__(self) -> int: 
        if self.mode == "test":
            return 1
        return len(self.outputs)

    def __getitem__(self, idx: int): 

        return self.outputs[idx]
    