# util functions about data

from scipy.stats import rankdata, iqr, trim_mean
from sklearn.metrics import f1_score, mean_squared_error
import numpy as np
from numpy import percentile


def adjust_predicts(score, label):
    """
    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    predict = score > 0.1
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    if not predict[j]:
                        predict[j] = True
                        latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    return predict.astype(np.float32)


#############

def get_attack_interval(attack):
    heads = []
    tails = []
    for i in range(len(attack)):
        if attack[i] == 1:
            if attack[i - 1] == 0:
                heads.append(i)

            if i < len(attack) - 1 and attack[i + 1] == 0:
                tails.append(i)
            elif i == len(attack) - 1:
                tails.append(i)
    res = []
    for i in range(len(heads)):
        res.append((heads[i], tails[i]))
    # print(heads, tails)
    return res


# calculate F1 scores
def eval_scores(scores, true_scores, th_steps, data_seq):
    padding_list = [0] * (len(true_scores) - len(scores))

    if len(padding_list) > 0:
        scores = padding_list + scores
    np.save(f'{data_seq}_scores.npy', scores)  # save anomaly scores

    scores_sorted = rankdata(scores, method='ordinal')
    th_vals = np.array(range(th_steps)) * 1.0 / th_steps
    fmeas = np.zeros(th_steps)
    thresholds = np.zeros(th_steps)
    adj_fmeas = np.zeros(th_steps)

    for i in range(th_steps):
        cur_pred = scores_sorted > th_vals[i] * len(scores)
        score_index = scores_sorted.tolist().index(int(th_vals[i] * len(scores) + 1))
        thresholds[i] = scores[score_index]
        adj_preds = list()

        f1 = f1_score(true_scores, cur_pred)
        np.save('cur_pred.npy', cur_pred)
        adj_pred = adjust_predicts(cur_pred, true_scores)
        np.save('adj_pred.npy', adj_pred)

        adj_preds.append(adj_pred)
        f1_adj = f1_score(true_scores, adj_pred)
        fmeas[i] = f1
        adj_fmeas[i] = f1_adj

    return fmeas.tolist(), thresholds.tolist(), adj_fmeas#.tolist()


def eval_mseloss(predicted, ground_truth):
    ground_truth_list = np.array(ground_truth)
    predicted_list = np.array(predicted)
    loss = mean_squared_error(predicted_list, ground_truth_list)
    return loss


def get_err_median_and_iqr(predicted, groundtruth):
    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    err_median = np.median(np_arr)
    err_iqr = iqr(np_arr)

    return err_median, err_iqr


def get_err_median_and_quantile(predicted, groundtruth, percentage):
    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))
    err_median = np.median(np_arr)
    err_delta = percentile(np_arr, int(percentage * 100)) - percentile(np_arr, int((1 - percentage) * 100))

    return err_median, err_delta


def get_err_mean_and_quantile(predicted, groundtruth, percentage):
    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    err_median = trim_mean(np_arr, percentage)
    # err_iqr = iqr(np_arr)
    err_delta = percentile(np_arr, int(percentage * 100)) - percentile(np_arr, int((1 - percentage) * 100))

    return err_median, err_delta


def get_err_mean_and_std(predicted, groundtruth):
    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    err_mean = np.mean(np_arr)
    err_std = np.std(np_arr)

    return err_mean, err_std


def get_f1_score(scores, gt, contamination):
    padding_list = [0] * (len(gt) - len(scores))
    # print(padding_list)

    threshold = percentile(scores, 100 * (1 - contamination))

    if len(padding_list) > 0:
        scores = padding_list + scores

    pred_labels = (scores > threshold).astype('int').ravel()

    return f1_score(gt, pred_labels)
