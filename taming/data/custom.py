import os
import numpy as np
import albumentations
import _pickle as cPickle
import random
from PIL import Image
from torch.utils.data import Dataset
import cv2

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex


class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example


class CustomTrain(CustomBase):
    def __init__(self, size, training_images_list_file):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)


class CustomTest(CustomBase):
    def __init__(self, size, test_images_list_file):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)



class VQBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example


class VQTrain(VQBase):
    def __init__(self, size, training_images_list_file, vq_path):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = VQPaths(paths=paths, vq_path=vq_path, size=size, random_crop=False, phase="train")


class VQTest(VQBase):
    def __init__(self, size, test_images_list_file, vq_path):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = VQPaths(paths=paths, vq_path=vq_path, size=size, random_crop=False, phase="test")
    

class VQPaths(ImagePaths):
    def __init__(self, paths, vq_path=None, size=None, random_crop=False, labels=None, phase="train"):
        super().__init__(paths, size, random_crop, labels)

        self.phase = phase
        with open(vq_path, "rb") as f:
            self.vq_tokens = cPickle.load(f)

        self.labels["t1_path_"] = []
        self.labels["t1ce_path_"] = []
        self.labels["t2_path_"] = []
        self.labels["flair_path_"] = []
        self.labels["seg_path_"] = []

        self.labels["t1_tokens_"] = []
        self.labels["t1ce_tokens_"] = []
        self.labels["t2_tokens_"] = []
        self.labels["flair_tokens_"] = []
        self.labels["seg_tokens_"] = []

        for path in self.labels["file_path_"]:
            if not os.path.exists(path.split(" ")[0]):
                self.labels["file_path_"].remove(path)

            if path not in self.vq_tokens:
                self.labels["file_path_"].remove(path)

            if 't1ce' in path:
                self.labels["t1ce_path_"].append(path)
                self.labels["t1ce_tokens_"].append(self.vq_tokens[path])
                path = path.replace("t1ce", "t1")
                self.labels["t1_path_"].append(path)
                self.labels["t1_tokens_"].append(self.vq_tokens[path])
                path = path.replace("t1", "t2")
                self.labels["t2_path_"].append(path)
                self.labels["t2_tokens_"].append(self.vq_tokens[path])
                path = path.replace("t2", "flair")
                self.labels["flair_path_"].append(path)
                self.labels["flair_tokens_"].append(self.vq_tokens[path])
                path = path.replace("flair", "seg")
                self.labels["seg_path_"].append(path)
                self.labels["seg_tokens_"].append(self.vq_tokens[path])

        self._length = len(self.labels["t1_path_"])

    def prepare_multilabel_masks(self, labels):
        original_values = [0, 255, 64, 127]
        mapped_values = [0, 1, 2, 3]
        labels = labels[:,:,:1]
        mapped_labels = labels.copy()
        for orig_val, mapped_val in zip(original_values, mapped_values):
            mapped_labels[labels == orig_val] = mapped_val

        tumor_core = (mapped_labels >= 1).astype(np.float32)
        enhancing_tumor = (mapped_labels >= 2).astype(np.float32)
        edema = (mapped_labels == 3).astype(np.float32)

        multilabel_masks = np.concatenate([tumor_core, enhancing_tumor, edema], axis=2)
        return multilabel_masks
    
    def preprocess_label(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        
        return image

    def __getitem__(self, i):
        example = dict()
        example["t1"] = self.preprocess_image(self.labels["t1_path_"][i].split(" ")[0])
        example["t1ce"] = self.preprocess_image(self.labels["t1ce_path_"][i].split(" ")[0])
        example["t2"] = self.preprocess_image(self.labels["t2_path_"][i].split(" ")[0])
        example["flair"] = self.preprocess_image(self.labels["flair_path_"][i].split(" ")[0])
        seg = self.preprocess_label(self.labels["seg_path_"][i].split(" ")[0])
        example["seg"] = self.prepare_multilabel_masks(seg)

        example["t1_tokens"] = self.labels["t1_tokens_"][i]
        example["t1ce_tokens"] = self.labels["t1ce_tokens_"][i]
        example["t2_tokens"] = self.labels["t2_tokens_"][i]
        example["flair_tokens"] = self.labels["flair_tokens_"][i]
        example["seg_tokens"] = self.labels["seg_tokens_"][i]

        return example


class VQBrains(Dataset):
    def __init__(self, brats2021_paths=None, brats2021_vq=None,
                       brats2023men_paths=None, brats2023men_vq=None,
                       ixi_paths=None, ixi_vq=None,
                       atlas2_paths=None, atlas2_vq=None,
                       mode="train"):
        super().__init__()

        brats2021_tr = VQBrain_tr(brats2021_paths, brats2021_vq, 
                            modality={"T1":"t1", "T1ce":"t1ce", "T2":"t2", "FLAIR":"flair"}, 
                            mode=mode)
        brats2021_seg = VQBrain_tr(brats2021_paths, brats2021_vq, 
                            roi={"Edema":"edema", "Enhancing-tumor":"et", "Tumor-core":"tc"},
                            mode=mode)
        brats2023men_tr = VQBrain_tr(brats2023men_paths, brats2023men_vq,
                                    modality={"T1":"t1", "T1ce":"t1ce", "T2":"t2", "FLAIR":"flair"}, 
                                    mode=mode)
        brats2023men_seg = VQBrain_tr(brats2023men_paths, brats2023men_vq,
                                    roi={"Hyperintensities":"hi", "Enhancing-tumor":"et", "Tumor-core":"tc"},
                                    mode=mode)
        ixi_tr = VQBrain_tr(ixi_paths, ixi_vq, modality={"T1":"t1", "T2":"t2", "PD":"pd"}, mode=mode)
        atlas2_seg = VQBrain_tr(atlas2_paths, atlas2_vq, roi={"Stroke":"seg"}, mode=mode)

        self.data = brats2021_tr + brats2021_seg + brats2023men_tr + brats2023men_seg + ixi_tr + atlas2_seg
        
    def __len__(self):
        return self.data.__len__()
    
    def __getitem__(self, i):
        return self.data.__getitem__(i)


class VQBrain_tr(ImagePaths):
    def __init__(self, paths, vq_path=None, modality=None, mode="train"):
        super().__init__(paths)

        with open(paths, 'r') as f:
            self.labels["file_path_"] = f.read().splitlines()

        with open(vq_path, "rb") as f:
            self.tokens = cPickle.load(f)

        self.labels["t1_path_"] = []
        self.modality = modality 
        self.mode = mode

        for path in self.labels["file_path_"]:
            if os.path.exists(path) and 't1' in path and path in self.tokens and 't1ce' not in path:    
                self.labels["t1_path_"].append(path)

        self._length = len(self.labels["t1_path_"])
    
    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict()

        if self.mode == "train" or "BraTS2021" not in self.labels["t1_path_"][i].split(" ")[0]:
            modality = list(self.modality.keys())
            source = random.choice(modality)
            modality.remove(source)
            target = random.choice(modality)
            mod_src = self.modality[source]
            mod_tgt = self.modality[target]
        else:
            source = "T1"
            target = "T1ce"
            mod_src = "t2"
            mod_tgt = "t1ce"

        source_path = self.labels["t1_path_"][i].split(" ")[0].replace("t1", mod_src)
        target_path = self.labels["t1_path_"][i].split(" ")[0].replace("t1", mod_tgt)

        example["source_img"] = self.preprocess_image(source_path)
        example["target_img"] = self.preprocess_image(target_path)
        example["target"] = target
        example["target_tokens"] = self.tokens[target_path]

        return example
    

class VQBrain_seg(ImagePaths):
    def __init__(self, paths, vq_path=None, roi=None, mode="train"):
        super().__init__(paths)

        with open(paths, 'r') as f:
            self.labels["file_path_"] = f.read().splitlines()

        with open(vq_path, "rb") as f:
            self.tokens = cPickle.load(f)

        self.labels["t1_path_"] = []
        self.roi = roi 
        self.mode = mode

        for path in self.labels["file_path_"]:
            if os.path.exists(path) and 't1' in path and path in self.tokens and 't1ce' not in path:    
                self.labels["t1_path_"].append(path)

        self._length = len(self.labels["t1_path_"])

    def prepare_masks(self, labels):
        labels = labels[:,:,:1]
        labels[labels == 255] = 1.0

        return labels
    
    def preprocess_label(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        
        return image
    
    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict()

        if self.mode == "train" or "BraTS2021" not in self.labels["t1_path_"][i].split(" ")[0]:
            seg = random.choice(list(self.roi.keys()))
            roi_seg = self.roi[seg]
        else:
            seg = "Edema"
            roi_seg = "edema"
        seg_path = self.labels["t1_path_"][i].split(" ")[0].replace("t1", roi_seg)
        example["target"] = seg
        example["seg"] = self.prepare_masks(self.preprocess_label(seg_path))
        example["seg_tokens"] = self.tokens[seg_path]

        return example
    

class VQBrains2_tr(Dataset):
    def __init__(self, i2i_tokens_path="i2i_tokens.pickle", 
                 mode="train"):
        super().__init__()

        with open(i2i_tokens_path, "rb") as f:
            self.i2i_tokens = cPickle.load(f)

        self.mode = mode
        if mode == "train":
            self.tokens = self.i2i_tokens
        else:
            self.tokens = self.i2i_tokens[:10]

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = (image/127.5 - 1.0).astype(np.float32)
        return image
        
    def __len__(self):
        if self.mode == "train":
            return len(self.tokens)
        else:
            return 10
    
    def __getitem__(self, i):
        example = dict()

        token = self.tokens[i]
        source = token["source"]
        target = token["target"]
        source_path = token["path"]
        
        target_path = source_path.replace(source, target)
        example["task"] = "tr"
        example["target_tokens"] = token["gen_image"]

        example["source"] = source
        example["target"] = target
        example["source_img"] = self.preprocess_image(source_path)
        example["target_img"] = self.preprocess_image(target_path) # translation
        

        return example

class VQBrains2_seg(Dataset):
    def __init__(self, seg_tokens_path="seg_tokens.pickle",
                 mode="train"):
        super().__init__()

        with open(seg_tokens_path, "rb") as f:
            self.seg_tokens = cPickle.load(f)
        
        self.mode = mode
        if mode == "train":
            self.tokens = self.seg_tokens
        else:
            self.tokens = self.seg_tokens[:10]
        self.mapping = {"edema":"edema", "enhancing_tumor":"et", "tumor_core":"tc",
                        "stroke":"seg", "hyperintensity":"hi"}

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = (image/127.5 - 1.0).astype(np.float32)
        return image
    
    def preprocess_label(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        
        # convert to gray scale
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image[image != 255] = 0
        image[image == 255] = 1
        
        return image
        
    def __len__(self):
        if self.mode == "train":
            return len(self.tokens)
        else:
            return 10
    
    def __getitem__(self, i):
        example = dict()

        token = self.tokens[i]
        source = token["source"]
        target = token["target"]
        source_path = token["path"]
        
        target_path = source_path.replace(source, self.mapping[target])
        example["task"] = "seg"

        example["target_tokens"] = token["gen_image"]

        example["source"] = source
        example["target"] = target
        example["source_img"] = self.preprocess_image(source_path)
        example["target_img"] = self.preprocess_label(target_path) # segmentation
        

        return example