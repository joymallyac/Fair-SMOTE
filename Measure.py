import numpy as np
import copy,math
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors


def get_counts(clf, x_train, y_train, x_test, y_test, test_df, biased_col, metric='aod'):
    
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    cnf_matrix = confusion_matrix(y_test, y_pred)



    TN, FP, FN, TP = confusion_matrix(y_test,y_pred).ravel()

    

    test_df_copy = copy.deepcopy(test_df)
    test_df_copy['current_pred_'  + biased_col] = y_pred

    test_df_copy['TP_' + biased_col + "_1"] = np.where((test_df_copy['Probability'] == 1) &
                                           (test_df_copy['current_pred_' + biased_col] == 1) &
                                           (test_df_copy[biased_col] == 1), 1, 0)

    test_df_copy['TN_' + biased_col + "_1"] = np.where((test_df_copy['Probability'] == 0) &
                                                  (test_df_copy['current_pred_' + biased_col] == 0) &
                                                  (test_df_copy[biased_col] == 1), 1, 0)

    test_df_copy['FN_' + biased_col + "_1"] = np.where((test_df_copy['Probability'] == 1) &
                                                  (test_df_copy['current_pred_' + biased_col] == 0) &
                                                  (test_df_copy[biased_col] == 1), 1, 0)

    test_df_copy['FP_' + biased_col + "_1"] = np.where((test_df_copy['Probability'] == 0) &
                                                  (test_df_copy['current_pred_' + biased_col] == 1) &
                                                  (test_df_copy[biased_col] == 1), 1, 0)

    test_df_copy['TP_' + biased_col + "_0"] = np.where((test_df_copy['Probability'] == 1) &
                                                  (test_df_copy['current_pred_' + biased_col] == 1) &
                                                  (test_df_copy[biased_col] == 0), 1, 0)

    test_df_copy['TN_' + biased_col + "_0"] = np.where((test_df_copy['Probability'] == 0) &
                                                  (test_df_copy['current_pred_' + biased_col] == 0) &
                                                  (test_df_copy[biased_col] == 0), 1, 0)

    test_df_copy['FN_' + biased_col + "_0"] = np.where((test_df_copy['Probability'] == 1) &
                                                  (test_df_copy['current_pred_' + biased_col] == 0) &
                                                  (test_df_copy[biased_col] == 0), 1, 0)

    test_df_copy['FP_' + biased_col + "_0"] = np.where((test_df_copy['Probability'] == 0) &
                                                  (test_df_copy['current_pred_' + biased_col] == 1) &
                                                  (test_df_copy[biased_col] == 0), 1, 0)

    a = test_df_copy['TP_' + biased_col + "_1"].sum()
    b = test_df_copy['TN_' + biased_col + "_1"].sum()
    c = test_df_copy['FN_' + biased_col + "_1"].sum()
    d = test_df_copy['FP_' + biased_col + "_1"].sum()
    e = test_df_copy['TP_' + biased_col + "_0"].sum()
    f = test_df_copy['TN_' + biased_col + "_0"].sum()
    g = test_df_copy['FN_' + biased_col + "_0"].sum()
    h = test_df_copy['FP_' + biased_col + "_0"].sum()

    
    if metric=='aod':
        return  calculate_average_odds_difference(a, b, c, d, e, f, g, h)
    elif metric=='eod':
        return calculate_equal_opportunity_difference(a, b, c, d, e, f, g, h)    
    elif metric=='recall':
        return calculate_recall(TP,FP,FN,TN)
    elif metric=='far':
        return calculate_far(TP,FP,FN,TN)
    elif metric=='precision':
        return calculate_precision(TP,FP,FN,TN)
    elif metric=='accuracy':
        return calculate_accuracy(TP,FP,FN,TN)
    elif metric=='F1':
        return calculate_F1(TP,FP,FN,TN)        
    elif metric=='TPR':
        return calculate_TPR_difference(a, b, c, d, e, f, g, h)
    elif metric=='FPR':
        return calculate_FPR_difference(a, b, c, d, e, f, g, h)
    elif metric == "DI":
    	return calculate_Disparate_Impact(a, b, c, d, e, f, g, h)
    elif metric == "SPD":
    	return calculate_SPD(a, b, c, d, e, f, g, h)



def calculate_average_odds_difference(TP_male , TN_male, FN_male,FP_male, TP_female , TN_female , FN_female,  FP_female):
    # TPR_male = TP_male/(TP_male+FN_male)
    # TPR_female = TP_female/(TP_female+FN_female)
    # FPR_male = FP_male/(FP_male+TN_male)
    # FPR_female = FP_female/(FP_female+TN_female)
    # average_odds_difference = abs(abs(TPR_male - TPR_female) + abs(FPR_male - FPR_female))/2
    FPR_diff = calculate_FPR_difference(TP_male , TN_male, FN_male,FP_male, TP_female , TN_female , FN_female,  FP_female)
    TPR_diff = calculate_TPR_difference(TP_male , TN_male, FN_male,FP_male, TP_female , TN_female , FN_female,  FP_female)
    average_odds_difference = (FPR_diff + TPR_diff)/2
    #print("average_odds_difference",average_odds_difference)
    return round(average_odds_difference,2)

def calculate_Disparate_Impact(TP_male , TN_male, FN_male,FP_male, TP_female , TN_female , FN_female,  FP_female):
    P_male = (TP_male + FP_male)/(TP_male + TN_male + FN_male + FP_male)
    P_female = (TP_female + FP_female)/(TP_female + TN_female + FN_female +  FP_female)    
    DI = (P_female/P_male)
    return round((1 - abs(DI)),2)

def calculate_SPD(TP_male , TN_male, FN_male,FP_male, TP_female , TN_female , FN_female,  FP_female):
    P_male = (TP_male + FP_male)/(TP_male + TN_male + FN_male + FP_male)
    P_female = (TP_female + FP_female) /(TP_female + TN_female + FN_female +  FP_female)       
    SPD = (P_female - P_male)
    return round(abs(SPD),2)


def calculate_equal_opportunity_difference(TP_male , TN_male, FN_male,FP_male, TP_female , TN_female , FN_female,  FP_female):
    # TPR_male = TP_male/(TP_male+FN_male)
    # TPR_female = TP_female/(TP_female+FN_female)    
    # equal_opportunity_difference = abs(TPR_male - TPR_female)
    #print("equal_opportunity_difference:",equal_opportunity_difference)
    return calculate_TPR_difference(TP_male , TN_male, FN_male,FP_male, TP_female , TN_female , FN_female,  FP_female)

def calculate_TPR_difference(TP_male , TN_male, FN_male,FP_male, TP_female , TN_female , FN_female,  FP_female):
    TPR_male = TP_male/(TP_male+FN_male)
    TPR_female = TP_female/(TP_female+FN_female)
    # print("TPR_male:",TPR_male,"TPR_female:",TPR_female)   
    diff = (TPR_male - TPR_female)
    return round(diff,2)

def calculate_FPR_difference(TP_male , TN_male, FN_male,FP_male, TP_female , TN_female , FN_female,  FP_female):
    FPR_male = FP_male/(FP_male+TN_male)
    FPR_female = FP_female/(FP_female+TN_female)
    # print("FPR_male:",FPR_male,"FPR_female:",FPR_female)    
    diff = (FPR_female - FPR_male)    
    return round(diff,2)


def calculate_recall(TP,FP,FN,TN):
    if (TP + FN) is not 0:
        recall = TP / (TP + FN)
    else:
        recall = 0
    return round(recall,2)

def calculate_far(TP,FP,FN,TN):
    if (FP + TN) is not 0:
        far = FP / (FP + TN)
    else:
        far = 0
    return round(far,2)

def calculate_precision(TP,FP,FN,TN):
    if (TP + FP) is not 0:
        prec = TP / (TP + FP)
    else:
        prec = 0
    return round(prec,2)

def calculate_F1(TP,FP,FN,TN):
    precision = calculate_precision(TP,FP,FN,TN)
    recall = calculate_recall(TP,FP,FN,TN)
    F1 = (2 * precision * recall)/(precision + recall)    
    return round(F1,2)

def calculate_accuracy(TP,FP,FN,TN):
    return round((TP + TN)/(TP + TN + FP + FN),2)

def consistency_score(X, y, n_neighbors=5):
        
    num_samples = X.shape[0]
    # y = y.values # Do it if it's not np array
    # learn a KNN on the features
    nbrs = NearestNeighbors(n_neighbors, algorithm='ball_tree').fit(X)
    _, indices = nbrs.kneighbors(X)

    # compute consistency score
    consistency = 0.0
    for i in range(num_samples):
        consistency += np.abs(y[i] - np.mean(y[indices[i]]))
    consistency = 1.0 - consistency/num_samples       
    return consistency


def measure_final_score(test_df, clf, X_train, y_train, X_test, y_test, biased_col, metric):
    df = copy.deepcopy(test_df)
    return get_counts(clf, X_train, y_train, X_test, y_test, df, biased_col, metric=metric)