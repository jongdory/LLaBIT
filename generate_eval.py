#### Start of environment setup ####

import os
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('model_path', type=Path,
                    help='Path to LLaBIT model checkpoint.')
parser.add_argument('vq_path', type=Path,
                    help='Path to Vector Quantized dataset pickle.')
parser.add_argument('output_root', type=Path,
                    help='Path to save result.')
parser.add_argument('--csv_path', type=Path, default="/store8/01.Database/01.Brain/15.2D/text-split-report.csv",
                    help='Path to JPG dataset.')
parser.add_argument('--task', type=str, default="report",
                    help='report, vga, i2i, seg')
parser.add_argument('--top_p', type=float, default=0.98)
parser.add_argument('--word_size', type=int, default=1,
                    help='Number of parallel processes.')
parser.add_argument('--rank', type=int, default=0,
                    help='Rank of current process.')
args = parser.parse_args()

N_PARALLEL = args.word_size
I_PARALLEL = args.rank

os.environ["CUDA_VISIBLE_DEVICES"] = str(I_PARALLEL)

#### End of environment setup ####
import pandas as pd
import pickle

from tqdm import tqdm

from training.generate import generate_response, load_model_tokenizer_for_generate
from training.brain_vq_dataset import sample_brain_vq_output_instruction, sample_brain_vq_input_instruction, sample_brain_vq_seg_instruction, VQ_TOKENIZER_LEN, VQ_VQ_LEN


def dicom_id_to_report_path(db, report_path, dicom_id: str):
    db_series = db.loc[dicom_id]
    subject_id = "p" + db_series["subject_id"]
    study_id = "s" + db_series["study_id"] + ".txt"
    subject_id_prefix = subject_id[:3]

    return report_path / Path("reports/files") / Path(subject_id_prefix) / Path(subject_id) / Path(study_id)
    
def load_report(db, report_path, dicom_id: str, parse_fun):
    report_path = dicom_id_to_report_path(db, report_path, dicom_id)
    with open(report_path, "r") as f:
        txt = f.readlines()
        
    return parse_fun(txt)
    
def parse_report_fi(txt: str) -> str:
    txt = " ".join([line.strip() for line in txt if line.strip() != ""])

    try:
        _, f_and_i = txt.split("FINDINGS:")
        try:
            f, i = f_and_i.strip().split("IMPRESSION:")
            f_and_i = f.strip() + " " + i.strip()
        except:
            f_and_i = f_and_i.strip()
    except:
        try:
            f_and_i = txt
            _, i = f_and_i.strip().split("IMPRESSION:")
            f_and_i = i.strip()
        except:
            raise ValueError

    return f_and_i
    
def parse_report_i(txt: str) -> str:
    txt = " ".join([line.strip() for line in txt if line.strip() != ""])
    
    try:
        _, impression = txt.strip().split("IMPRESSION:")
    except:
        raise ValueError
    
    return impression.strip()

def generate_vq_response(input_text, model, tokenizer, instruction_text, top_p):
    response_vq = None
    count = 0
    while response_vq is None or len(response_vq) != VQ_VQ_LEN:
        if count > 0:
            print("warning: retrying vq-gen")
        _, response_vq = generate_response((instruction_text, input_text), model=model, tokenizer=tokenizer, max_new_tokens=300, top_p=top_p)
        count += 1
    
    return response_vq

if __name__ == "__main__":

    chk_num = str(args.model_path).split("-")[-1]
    RESULT_PATH = args.output_root / Path(f"llabit__eval_{chk_num}_{args.task}_{args.top_p}.pickle")
    PARSE_FUNCTION = parse_report_i

    print(f"Result will be saved to {RESULT_PATH}.")

    db = pd.read_csv(args.csv_path, index_col="path", dtype=str)
    with open(args.vq_path, "rb") as f:
        db_vq = pickle.load(f)
    db.sort_index(inplace=True)
    dataset = []
    def get_raw_image(path, dataname):
        """Get VQ encoded image data."""
        if dataname == "ATLAS_2":
            path = path.replace("ATLAS_2", "ATLAS_2/Training")
        
        try:
            return [vq_elem + VQ_TOKENIZER_LEN for vq_elem in db_vq[path]]
        except:
            return None
    
    def create_data_entry(path, id, raw_image, source, target=None):
        """Create dataset entry."""
        return {
            "path": path, 
            "id": id, 
            "raw_image": raw_image,
            "source": source,
            "target": target,
            "gen_report": None, 
            "gen_image": None
        }
    
    def add_modality_combinations(dataset, path, id, raw_image, source_mod, modality_list):
        """Add all target modality combinations excluding source modality."""
        for target in modality_list:
            if target != source_mod:
                dataset.append(create_data_entry(path, id, raw_image, source_mod, target))
    
    # Dataset construction
    for path, id, mod, dataname in zip(db.index, db["id"], db["mod"], db["dataset"]):
        raw_image = get_raw_image(path, dataname)
        if raw_image is None:
            continue
            
        if args.task == "i2i":
            if dataname == "ATLAS_2":
                # ATLAS_2: only t1 -> t2 conversion
                dataset.append(create_data_entry(path, id, raw_image, "t1", "t2"))
            elif dataname == "IXI":
                # IXI: cross-conversion between t1, t2, pd
                modality_list = ["t1", "t2", "pd"]
                add_modality_combinations(dataset, path, id, raw_image, mod, modality_list)
            else:
                # BraTS: cross-conversion between t1, t2, t1ce, flair
                modality_list = ["t1", "t2", "t1ce", "flair"]
                add_modality_combinations(dataset, path, id, raw_image, mod, modality_list)
        else:
            # report, vqa, seg tasks
            if dataname == "IXI":
                continue
            dataset.append(create_data_entry(path, id, raw_image, mod))
    
    dataset = dataset[I_PARALLEL::N_PARALLEL]

    close_end_inst = "This is a close-ended question, so choose only one answer from the viewpoint and answer with a short answer."

    def generate_report(data, model, tokenizer):
        """Generate report."""
        instruction_text = sample_brain_vq_input_instruction()
        response, _ = generate_response((instruction_text, data["raw_image"]), model=model, tokenizer=tokenizer, max_new_tokens=512)
        data["gen_report"] = response
    
    def generate_vqa_responses(data, model, tokenizer):
        """Generate VQA responses."""
        questions = ["abnorm_question", "modality_question", "plane_question"]
        response_keys = ["gen_response_abnomality", "gen_response_modality", "gen_response_plane"]
        
        for question_key, response_key in zip(questions, response_keys):
            instruction_text = data[question_key] + "\n" + close_end_inst + "\n"
            response, _ = generate_response((instruction_text, data["raw_image"]), model=model, tokenizer=tokenizer, max_new_tokens=256)
            data[response_key] = response
    
    def generate_i2i(data, model, tokenizer, top_p):
        """Generate image-to-image conversion."""
        instruction_text = sample_brain_vq_output_instruction()
        instruction_text = instruction_text.replace("<output>", data["target"]).replace("<input>", data["source"])
        response_vq = generate_vq_response(data["raw_image"], model, tokenizer, instruction_text, top_p=top_p)
        data["gen_image"] = response_vq
    
    def generate_segmentation(data, model, tokenizer, top_p):
        """Generate segmentation."""
        path = data["path"]
        
        if "ATLAS" in path:
            # ATLAS: stroke segmentation
            instruction_text = sample_brain_vq_seg_instruction().replace("<seg>", "stroke")
            response_vq = generate_vq_response(data["raw_image"], model, tokenizer, instruction_text, top_p=top_p)
            data["gen_response_stroke"] = response_vq
        else:
            # BraTS: tumor-related segmentation
            seg_tasks = [
                ("enhancing_tumor", "gen_response_enhancing_tumor"),
                ("tumor_core", "gen_response_tumor_core")
            ]
            
            # Check if MEN dataset to determine additional segmentation
            if "MEN" in path:
                seg_tasks.append(("hyperintensity", "gen_response_hyperintensity"))
            else:
                seg_tasks.append(("edema", "gen_response_edema"))
            
            for seg_type, response_key in seg_tasks:
                instruction_text = sample_brain_vq_seg_instruction().replace("<seg>", seg_type)
                response_vq = generate_vq_response(data["raw_image"], model, tokenizer, instruction_text, top_p=top_p)
                data[response_key] = response_vq

    # Model loading and generation execution
    model, tokenizer = load_model_tokenizer_for_generate(args.model_path)
    assert len(tokenizer) == VQ_TOKENIZER_LEN
    
    for data in tqdm(dataset, colour="green"):
        if args.task == "report":
            generate_report(data, model, tokenizer)
        elif args.task == "vqa":
            generate_vqa_responses(data, model, tokenizer)
        elif args.task == "i2i":
            generate_i2i(data, model, tokenizer, args.top_p)
        else:  # seg
            generate_segmentation(data, model, tokenizer, args.top_p)
        
    args.output_root.mkdir(parents=True, exist_ok=True)

    with open(RESULT_PATH, "wb") as f:
        pickle.dump(dataset, f)