from metrics.f1_score_f1_pa import *
from metrics.fc_score import *
from metrics.precision_at_k import *
from metrics.customizable_f1_score import *
from metrics.AUC import *
from metrics.Matthews_correlation_coefficient import *
from metrics.affiliation.generics import convert_vector_to_events
from metrics.affiliation.metrics import pr_from_events
from metrics.vus.models.feature import Window
from metrics.vus.metrics import get_range_vus_roc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import f1_score

def combine_all_evaluation_scores(pred, gt, anomaly_scores):
    events_pred = convert_vector_to_events(pred) 
    events_gt = convert_vector_to_events(gt)
    Trange = (0, len(pred))
    affiliation = pr_from_events(events_pred, events_gt, Trange)
    true_events = get_events(pred)
    accuracy = accuracy_score(gt, pred)
    precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                          average='binary')
    pa_accuracy, pa_precision, pa_recall, pa_f_score = get_adjust_F1PA(pred, gt)
    
    MCC_score = MCC(pred, gt)
    vus_results = get_range_vus_roc(pred, gt, 100) # default slidingWindow = 100
    
    score_list_simple = {
                    "accuracy":accuracy,
                    "precision":precision,
                    "recall":recall,
                    "f_score":f_score,
                    "pa_accuracy":pa_accuracy, 
                    "pa_precision":pa_precision, 
                    "pa_recall":pa_recall, 
                    "pa_f_score":pa_f_score,
                    "MCC_score":MCC_score, 
                    "Affiliation precision": affiliation['precision'], 
                    "Affiliation recall": affiliation['recall'],
                    "R_AUC_ROC": vus_results["R_AUC_ROC"], 
                    "R_AUC_PR": vus_results["R_AUC_PR"],
                    "VUS_ROC": vus_results["VUS_ROC"],
                    "VUS_PR": vus_results["VUS_PR"]
                  }
    
    # return score_list, score_list_simple
    return score_list_simple

def calculate_f1(predictions, gt):
    return f1_score(gt, predictions)

# 二分法查找最佳阈值以获得最高的F1分数
# def find_best_f1(test_energy, gt, num_thresholds=100):
#     best_threshold = 0.0
#     best_f1 = 0.0

#     thresholds = np.linspace(0, 1, num_thresholds)
#     for threshold in thresholds:
#         # 预测标签
#         predictions = (test_energy >= threshold).astype(int)
        
#         # 计算F1分数
#         current_f1 = calculate_f1(predictions, gt)
        
#         if current_f1 > best_f1:
#             best_f1 = current_f1
#             best_threshold = threshold
    
#     return best_threshold, best_f1

def find_best_f1(test_energy, gt, num_iterations=1000, step_size=0.05):
    best_threshold = np.random.rand()  # 随机初始化阈值
    best_f1 = calculate_f1((test_energy >= best_threshold).astype(int), gt)

    for _ in range(num_iterations):
        # 随机生成一个新的阈值
        new_threshold = best_threshold + np.random.uniform(-step_size, step_size)
        new_threshold = np.clip(new_threshold, 0, 1)  # 确保阈值在[0, 1]范围内
        
        # 预测标签
        predictions = (test_energy >= new_threshold).astype(int)
        
        # 计算新的 F1 分数
        current_f1 = calculate_f1(predictions, gt)
        
        # 更新最佳阈值和 F1 分数
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = new_threshold
    
    return best_threshold, best_f1


if __name__ == '__main__':
    pred = np.load("data/events_pred_MSL.npy")+0
    gt = np.load("data/events_gt_MSL.npy")+0
    anomaly_scores = np.load("data/events_scores_MSL.npy")
    print(len(pred), max(anomaly_scores), min(anomaly_scores))
    score_list_simple = combine_all_evaluation_scores(pred, gt, anomaly_scores)

    for key, value in score_list_simple.items():
        print('{0:21} :{1:10f}'.format(key, value))