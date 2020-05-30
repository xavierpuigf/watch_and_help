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
files = [
        'test_json_output_graph_multi_classifer_sort_insamelen_hid512_larger_largerv2_smallerv2_tranf_dp0_lstmavg_h4l1-new_single_task.p',
        'test_json_output_graph_sort_avg_insamelen_hid512_larger_largerv2_smallerv2_tranf_dp0_lstmavg_h2l1_newtest.p'
]

multiples = [False, True]
it = 0
for it in range(2):
    file_name = files[it]
    multiple = multiples[it]

    with open(file_name, 'rb') as f:
        ct = pkl.load(f)
    cand_files = ct.keys()
    pdb.set_trace()
    if multiple:

        remove_files = [
                "logs_agent_77_read_book_0.pik"
                "logs_agent_155_read_book_0.pik"
                "logs_agent_124_put_dishwasher_0.pik"
                "logs_agent_67_read_book_0.pik"
                "logs_agent_79_read_book_0.pik"
                "logs_agent_135_put_dishwasher_0.pik"
                "logs_agent_94_put_dishwasher_0.pik"]
        cand_files = [x for x in cand_files if x not in remove_files]


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
    print('Precision. {}, Recall. {}, F1. {}, {}'.format(precision, recall, f1, correct*1./len(cand_files)))
