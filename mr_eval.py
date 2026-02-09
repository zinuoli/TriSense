import json
import os
import time
import sys
import argparse
import pdb

def read_json(path):
    with open(path, "r") as fin:
        datas = json.load(fin)
    return datas


def iou(A, B):
    # print(A, B)
    if len(A) == 1:
        A = [A[0], A[0]]
    if len(B) == 1:
        B = [B[0], B[0]]
    max0 = max((A[0]), (B[0]))
    min0 = min((A[0]), (B[0]))
    max1 = max((A[1]), (B[1]))
    min1 = min((A[1]), (B[1]))
    return max(min1 - max0, 0) / (max1 - min0)


def toSec(timeStr):
    t = time.strptime(timeStr, "%H:%M:%S")
    return t.tm_hour * 3600 + t.tm_min * 60 + t.tm_sec

def captiondata_modify(steps):
    modify_data = {}
    for i, step in enumerate(steps[0]):
        for key in step["step"].keys():
            name = step["step"][key]["query_idx"]
            modify_data[name] = [[step['step'][key]["startime"], step['step'][key]["endtime"]]]
        
    return modify_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, default="rebuttal_slot/fmt_avs_slot32_f64_result_8.json")
    parser.add_argument('--gt_file', type=str, default='/mnt/chenyue/chenyue/zinuo/jsons/rebuttal/test_500.json')
    # parser.add_argument("--pred_file", type=str, default="eval_results_sc/fmt_debug_avs_sc_f64_result_14.json")
    # parser.add_argument('--gt_file', type=str, default='/mnt/chenyue/chenyue/zinuo/jsons/test_1.1w.json')
    # parser.add_argument('--pred_file', type=str, default='eval_results_vision/fmt_debug_charades_full_f64_result_0.json')
    # parser.add_argument('--gt_file', type=str, default='/mnt/chenyue/chenyue/zinuo/jsons/test_charades.json')
    parser.add_argument('--sample', action='store_true', default=False)
    args = parser.parse_args()
    '''
    {
        "query_idx": [start_time, end_time],
        ...
    }
    '''
    answer = read_json(args.gt_file)
    gt_timestamps = {}
    for jterm in answer:
        # gt_timestamps[jterm["id"]] = jterm["timestamp"]
        gt_timestamps[jterm["id"]] = jterm["times"][0]
    
        
        
    submission = read_json(args.pred_file)
    pred_timestamps = {}
    num = 0
    for jterm in submission:
        qid = jterm["id"]
        if 'timestamps' not in jterm or len(jterm["timestamps"]) == 0:
            pred_timestamps[int(qid)] = [0, 0]
            continue
        # if submission[qid]['vid'] != answer[int(qid)]['image_id'] or submission[qid]['query'].strip() != answer[int(qid)]['caption'].strip():
        #     print(f"{submission[qid]['vid']}\n{answer[int(qid)]['image_id']}\n{submission[qid]['query']}\n{answer[int(qid)]['caption']}")
        num += 1
        pred_timestamps[int(qid)] = jterm["timestamps"][0]

    if args.sample:
        new = {}
        for qid in pred_timestamps.keys():
            new[qid] = gt_timestamps[qid]
        gt_timestamps = new
    # num = len(gt_timestamps)
    print(f"# pred video timestamps {len(pred_timestamps)}; # gt video timestamps {len(gt_timestamps)}")
    # assert len(gt_timestamps) == len(pred_timestamps)
    Result = {0: 0, 0.3:0, 0.5:0, 0.7:0, 'miou': 0}
    for key in gt_timestamps.keys():
        if key not in pred_timestamps or len(pred_timestamps[key]) < 1:
            continue
        temp = iou(gt_timestamps[key], pred_timestamps[key])
        for c_iou in [0.0, 0.3, 0.5, 0.7]:
            if(temp >= c_iou):
                Result[c_iou] = Result[c_iou] + 1
        Result['miou'] += temp
    print(num, sum(Result.values()), Result)
    print("IOU 0.3: {0}\nIOU 0.5: {1}\nIOU 0.7: {2}\n mIOU: {3}".format(Result[0.3]*100/num, Result[0.5]*100/num, Result[0.7]*100/num, Result['miou']*100/num))