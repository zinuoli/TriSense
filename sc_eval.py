import json
import argparse
import numpy as np
from tqdm import tqdm

# Import evaluation metrics from pycocoevalcap
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

def read_json(path):
    with open(path, "r") as fin:
        datas = json.load(fin)
    return datas

def extract_gt_captions(gt_data, type):
    gt_captions = {}
    for item in gt_data:
        item_id = item["id"]
        if "conversations" in item and len(item["conversations"]) > 1:
            caption = item["conversations"][type].get("value", "")
            gt_captions[item_id] = caption
        if "sentences" in item and len(item["sentences"]) > 1:
            caption = item["sentences"][0]
            gt_captions[item_id] = caption
    return gt_captions

def extract_pred_captions(pred_data):
    pred_captions = {}
    for item in pred_data:
        item_id = item["id"]
        if "generated_cap" in item:
            caption = item["generated_cap"]
            pred_captions[item_id] = caption
        elif "captions" in item:
            caption = item["captions"]
            pred_captions[item_id] = caption
    return pred_captions

def preprocess_text(text):
    if type(text) == list:
        text = text[0]
    """Simple text preprocessing"""
    return text.lower().strip()

def calculate_metrics(gt_captions, pred_captions):
    """Calculate all evaluation metrics using pycocoevalcap"""
    # Ensure text is preprocessed
    processed_gt = {k: preprocess_text(v) for k, v in gt_captions.items()}
    processed_pred = {k: preprocess_text(v) for k, v in pred_captions.items()}
    
    # Prepare data in pycocoevalcap format
    gts = {}
    res = {}
    
    for key in processed_gt.keys():
        gts[key] = [{'caption': processed_gt[key]}]
        res[key] = [{'caption': processed_pred[key]}]
    
    # Process reference and candidate text using PTB tokenizer
    tokenizer = PTBTokenizer()
    gts_tokens = tokenizer.tokenize(gts)
    res_tokens = tokenizer.tokenize(res)
    
    # Initialize evaluators
    scorers = [
        (Bleu(4), "Bleu_4"),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    
    # Calculate all metrics
    metrics = {}
    print("Calculating evaluation metrics...")
    for scorer, method in tqdm(scorers):
        score, scores = scorer.compute_score(gts_tokens, res_tokens)
        if method == "Bleu_4":
            # Bleu returns multiple values, take Bleu-4 score (index 3)
            metrics["BLEU-4"] = score[3]
        else:
            metrics[method] = score
    
    return metrics

if __name__ == "__main__":
    # Check Java environment (METEOR requires Java)
    import os
    java_path = os.environ.get('JAVA_HOME')
    if not java_path:
        print("Warning: JAVA_HOME environment variable is not set, METEOR evaluation may fail")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, default="rebuttal_slot/fmt_avs_slot32_f64_result_0.json")
    parser.add_argument('--gt_file', type=str, default='/mnt/chenyue/chenyue/zinuo/jsons/rebuttal/test_500.json')
    args = parser.parse_args()
    
    # Read files
    gt_data = read_json(args.gt_file)
    pred_data = read_json(args.pred_file)
    
    # Extract captions
    gt_captions = extract_gt_captions(gt_data, type=5)
    # print(gt_captions)
    pred_captions = extract_pred_captions(pred_data)
    
    # Ensure all predictions have corresponding ground truth annotations
    common_ids = set(gt_captions.keys()) & set(pred_captions.keys())
    filtered_gt = {k: gt_captions[k] for k in common_ids}
    filtered_pred = {k: pred_captions[k] for k in common_ids}
    
    print(f"Number of valid samples: {len(common_ids)}")
    
    # Calculate metrics
    metrics = calculate_metrics(filtered_gt, filtered_pred)
    
    # Print results
    print(f"BLEU-4: {metrics['BLEU-4']*100:.4f}")
    print(f"METEOR: {metrics['METEOR']*100:.4f}")
    print(f"ROUGE_L: {metrics['ROUGE_L']*100:.4f}")
    print(f"CIDEr: {metrics['CIDEr']*100:.4f}")