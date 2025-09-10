import os
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
import numpy as np
import json

from pathlib import Path
from PIL import Image
import pickle
import random
from typing import List 

from .consts import RESPONSE_KEY_NL

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

VQ_SYNTH_INSTRUCTION_LIST = [
    "Transform the brain MR image from <input> to <output>, ensuring the output reflects the characteristics of <output>.",
    "Convert the input brain MR image from <input> into a brain MR image in <output>.",
    "Use the brain MR image in <input> to generate its corresponding image in <output>.",
    "Generate a brain MR image in <output> based on the input from <input>.",
    "Create a <output> brain MR image that mirrors the characteristics of the <input>.",
    "Produce a brain MR image in <output> using the provided <input> image.",
    "Transform the input brain MR image from <input> into the corresponding image in <output>.",
    "Based on the provided brain MR image in <input>, create the equivalent image in <output>.",
    "Generate a brain MR image in <output> that aligns with the features of the <input> image.",
    "Synthesize a brain MR image in <output> using the input image from <input>.",
]

VQ_SEG_INSTRUCTION_LIST = [
    "Generate a segmentation map for <seg> in a given brain MR image.",
    "Create a segmentation of <seg> in the brain MR image that highlights key structures.",
    "Produce a segmented brain MR image identifying <seg>-related regions and structures.",
    "Use the provided brain MR image to create the corresponding <seg> segmentation map.",
    "Generate a brain MR image segmentation that accurately delineates regions of <seg>.",
    "Based on the input brain MR image, produce a detailed <seg> segmentation map.",
    "Use the input brain MR image to generate a segmentation highlighting areas of <seg>.",
    "Create a segmentation of <seg> in the brain MR image corresponding to the input image.",
    "Produce a brain MR image segmentation map that identifies regions of <seg>.",
    "Generate a <seg> segmentation map from the given brain MR image to enhance visualization.",
]

VQ_INPUT_INSTRUCTION_LIST = [
    "Generate free-text radiology reports for the provided brain MR images.",
    "Use the provided brain MR images to create corresponding free-text radiology reports.",
    "Based on the provided brain MR images, produce free-text radiology reports.",
    "Create free-text radiology reports that correspond to the provided brain MR images.",
    "Utilize the provided brain MR images to generate corresponding free-text radiology reports.",
    "Generate free-text radiology reports in accordance with the provided brain MR images.",
    "Use the provided brain MR images to create accurate free-text radiology reports.",
    "Produce free-text radiology reports that match the provided brain MR images.",
    "Create free-text radiology reports that are consistent with the provided brain MR images.",
    "Utilize the provided brain MR images to generate comprehensive free-text radiology reports.",
]

VQ_IN_REPLACE_TEMPLATE = "<input>"
VQ_OUT_REPLACE_TEMPLATE = "<output>"
VQ_SEG_REPLACE_TEMPLATE = "<seg>"

VQ_CODE_BOOK_SIZE = 1024
VQ_VQ_LEN = 144
VQ_TOKENIZER_LEN = 50281 #128256 #50281


class BrainVqDataset(Dataset): 
    def __init__(self, tokenizer_len: int, mode: str):
        assert tokenizer_len == VQ_TOKENIZER_LEN
        self.mode = mode
        self.tokenizer_len = tokenizer_len

        with open("total_report.json", "r") as f:
            brain_info = json.load(f)

        with open("total_f16.pkl", "rb") as f:
            self.brain_toks  = pickle.load(f)

        self.outputs = []
        for row in brain_info:
            image_path = row["path"]
            dataset = row["dataset"]
            modalities, segs = self.get_modalities_and_segs(dataset, row)
            brain_vq = self.brain_toks[image_path]
            brain_vq_shifted = [x + tokenizer_len for x in brain_vq]

            instruction = random.choice(VQ_INPUT_INSTRUCTION_LIST)
            self.append_output(row["report"], brain_vq_shifted, None, None, "input", instruction)

            modality_1 = row["modality"]
            if modality_1 == "t1":
                if dataset == "ATLAS_2":
                    if not segs: continue  
                    instruction, brain_vq_seg = self.get_info_for_segmentation(modality_1, segs, image_path)
                    self.append_output(None, brain_vq_shifted, None, brain_vq_seg, "seg", instruction)
                elif dataset == "IXI":
                    for target in ["t2"]:
                        instruction, brain_vq_output = self.get_info_for_translation(modality_1, [target], image_path)
                        self.append_output(None, brain_vq_shifted, brain_vq_output, None, "output", instruction)
                else:
                    for target in ["t2"]:
                        instruction, brain_vq_output = self.get_info_for_translation(modality_1, [target], image_path)
                        self.append_output(None, brain_vq_shifted, brain_vq_output, None, "output", instruction)
            elif modality_1 == "t1ce":
                if not segs: continue
                for seg_key, seg_value in segs.items():
                    instruction, brain_vq_seg = self.get_info_for_segmentation(modality_1, {seg_key: seg_value}, image_path)
                    self.append_output(None, brain_vq_shifted, None, brain_vq_seg, "seg", instruction)
            else:
                continue

    def append_output(self, report, brain_vq_shifted, brain_vq_output, brain_vq_seg, io_type, instruction):
        self.outputs.append({
            "report": report,
            "brain_vq_shifted": brain_vq_shifted,
            "brain_vq_output": brain_vq_output,
            "brain_vq_seg": brain_vq_seg,
            "io_type": io_type,
            "instruction": instruction
        })

    def get_info_for_translation(self, input, modalities, image_path):
        output = random.choice(modalities)
        instruction = random.choice(VQ_SYNTH_INSTRUCTION_LIST)
        instruction = instruction.replace("<input>", input).replace("<output>", output)
        target_path = image_path.replace(input, output)
        target_brain_vq = self.brain_toks[target_path]
        brain_vq_output = [x + self.tokenizer_len for x in target_brain_vq]

        return instruction, brain_vq_output
    
    def get_info_for_segmentation(self, input, segs, image_path):
        instruction = random.choice(VQ_SEG_INSTRUCTION_LIST)
        roi_key = random.choice(list(segs.keys()))
        instruction = instruction.replace("<seg>", roi_key)
        target_path = image_path.replace(input, segs[roi_key])
        target_brain_vq = self.brain_toks[target_path]
        brain_vq_seg = [x + self.tokenizer_len for x in target_brain_vq]

        return instruction, brain_vq_seg

    def get_modalities_and_segs(self, dataset, row):
        if dataset == "ATLAS_2":
            modalities = ["t1"]
            segs = {"stroke":"seg"}
            if row["stroke"] == 0: segs.pop("stroke")
        elif dataset == "IXI":
            modalities = ["t1", "t2", "pd"]
            segs = {}
        else:
            modalities = ["t1", "t2", "t1ce", "flair"]
            if dataset == "BraTS2021":
                segs = {"edema":"edema", "enhancing_tumor":"et","tumor_core":"tc"}
                if row["edema"] == 0: segs.pop("edema")
                if row["enhancing_tumor"] == 0: segs.pop("enhancing_tumor")
                if row["tumor_core"] == 0: segs.pop("tumor_core")
            else:
                segs = {"hyperintensity":"hi", "enhancing_tumor":"et","tumor_core":"tc"}
                if row["hyperintensity"] == 0: segs.pop("hyperintensity")
                if row["enhancing_tumor"] == 0: segs.pop("enhancing_tumor")
                if row["tumor_core"] == 0: segs.pop("tumor_core")

        return modalities, segs

    def __len__(self) -> int: 
        if self.mode == "test":
            return 100
        return len(self.outputs)

    def __getitem__(self, idx: int):
        return self.outputs[idx]

    
def sample_brain_vq_output_instruction():
    return random.choice(VQ_SYNTH_INSTRUCTION_LIST)

def sample_brain_vq_input_instruction():
    return random.choice(VQ_INPUT_INSTRUCTION_LIST)

def sample_brain_vq_seg_instruction():
    return random.choice(VQ_SEG_INSTRUCTION_LIST)

def _find_vq_replace_token_idx(input_ids: List[int], vq_replace_token_ids: List[int]):
    assert len(vq_replace_token_ids) == 3
    
    for i in range(len(input_ids)):
        if input_ids[i:i+len(vq_replace_token_ids)] == vq_replace_token_ids:
            return i
    return None

def get_inject_vq_fun(tokenizer):
    
    def inject_vq(input_ids: List[int], brain_vq_shifted: List[int], template: str) -> List[int]:
        replace_tokens = tokenizer(template)['input_ids']

        if brain_vq_shifted is None: return input_ids
        
        assert len(brain_vq_shifted) == VQ_VQ_LEN
        assert max(brain_vq_shifted) >= VQ_TOKENIZER_LEN

        first_idx = _find_vq_replace_token_idx(input_ids, replace_tokens)
        second_idx = _find_vq_replace_token_idx(input_ids[first_idx+1:], replace_tokens)

        assert first_idx is not None
        assert second_idx is None

        return input_ids[:first_idx+1] + brain_vq_shifted + input_ids[first_idx+2:]
    
    return inject_vq

def get_extract_vq_fun(tokenizer, template: str):
    assert len(tokenizer) == VQ_TOKENIZER_LEN
    img_token_id = tokenizer(template)['input_ids']
    response_token_id = tokenizer(RESPONSE_KEY_NL)['input_ids']
    assert len(img_token_id) == 1
    assert len(response_token_id) == 1
    img_token_id = img_token_id[0]
    response_token_id = response_token_id[0]

    def extract_vq(input_ids: List[int]):
        sequence = input_ids.clone().flatten().cpu().numpy()
        reponse_start = np.where(sequence == response_token_id)[0][0] + 1

        is_vq = sequence >= VQ_TOKENIZER_LEN
        if np.any(is_vq):
            if len(np.where(is_vq)[0]) <= VQ_VQ_LEN:
                return None
            vq_start = np.where(is_vq)[0][VQ_VQ_LEN]
            if vq_start >= reponse_start:
                vq = sequence[is_vq] - VQ_TOKENIZER_LEN
                vq = vq[VQ_VQ_LEN:]
                if len(vq) == VQ_VQ_LEN:
                    vq = vq.tolist()
                    input_ids[..., vq_start] = img_token_id
                    return vq
                else: 
                    print(f"VQ token found but not of length {VQ_VQ_LEN}: {len(vq)}")
                    return None
            else:
                return None
        else:
            return None
    
    return extract_vq



if __name__ == "__main__":
    pass
