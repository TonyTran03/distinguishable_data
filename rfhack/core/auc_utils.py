#auc_utils.py
from sklearn.metrics import roc_auc_score
from scipy.stats import norm
import numpy 
 
 
def auc(y_true, y_score):
    return roc_auc_score(y_true, y_score)
 
 
def sep_from_auc(target_auc):
    return numpy.sqrt(2) * norm.ppf(target_auc)
 
 
def split_xy(df, target_col='target'):
    return df.drop(columns=[target_col]), df[target_col]
 