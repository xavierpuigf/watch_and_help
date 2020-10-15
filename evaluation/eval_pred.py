import pickle as pkl
import  pdb
from collections import Counter

def substr(x, y):
    # remove list without duplicates
    xn = [elem for elem in x]
    for yelem in y:
        if yelem in xn:
            xn.remove(yelem)
    return xn
def get_prec_recall_measures(gt, pred):
    curr_pred = substr(pred, gt)
    curr_gt = substr(gt, pred)
    tp =  len(pred) - len(curr_pred)
    fp = len(curr_pred)
    fn = len(curr_gt) 
    # if fp > 0. and tp > 0.:
    #    pdb.set_trace()
    return tp,fp,fn
files = ['../dataset/watch_pred.p']

it = 0
for it in range(len(files)):
    file_name = files[it]

    with open(file_name, 'rb') as f:
        ct = pkl.load(f)
    cand_files = ct.keys()


    tps, fps, fns, correct  = 0, 0, 0, 0
    for cand_file in cand_files:
        content_pred = ct[cand_file]
        gt = [x for x in content_pred['ground_truth'] if x not in [None, 'None']]
        pred = [x for x in content_pred['prediction'] if x not in [None, 'None']]
        tp, fp, fn = get_prec_recall_measures(gt, pred)
        tps += tp
        fps += fp
        fns += fn
        if (fp + fn) == 0:
            correct += 1
    precision = 1.0 * tps / (tps + fps)
    recall = 1.0 * tps / (tps + fns)
    f1 = 2 * precision * recall / (precision + recall)
    print('Precision. {}, Recall. {}, F1. {}, Accuracy: {}'.format(precision, recall, f1, correct*1./len(cand_files)))
