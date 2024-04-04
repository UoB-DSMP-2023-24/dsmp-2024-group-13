from filtering_functions import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import label_binarize
from scipy import interp
from sklearn.base import clone
from sklearn.decomposition import PCA, KernelPCA
from sklearn.cross_decomposition import CCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import warnings


# Function to generate k-mers from a single sequence
def generate_kmers(sequence, k):
    return [sequence[i:i+k] for i in range(len(sequence)-k+1)]



def create_features_matrix(df, include_alpha=True, include_beta=True, alpha_col='cdr3.alpha', beta_col='cdr3.beta', label_col='antigen.epitope', k=3):
    # Filter rows where label is missing
    filtered_df = df.dropna(subset=[label_col])
    
    # Initialize documents for CountVectorizer and k-mer count dictionary
    kmer_docs = []
    kmer_count_dict = {}
    
    # Process sequences based on inclusion flags
    for _, row in filtered_df.iterrows():
        kmers = []
        if include_alpha and pd.notna(row[alpha_col]):
            alpha_seq = row[alpha_col]
            kmers += generate_kmers(alpha_seq, k)
        if include_beta and pd.notna(row[beta_col]):
            beta_seq = row[beta_col]
            kmers += generate_kmers(beta_seq, k)
        
        # Concatenate k-mers into a single string for vectorization
        kmer_docs.append(' '.join(kmers))
        
        # Count occurrences of each k-mer
        for kmer in kmers:
            kmer_count_dict[kmer] = kmer_count_dict.get(kmer, 0) + 1
    
    # Vectorize k-mer documents into a feature matrix
    vectorizer = CountVectorizer(analyzer='word', token_pattern=r"(?u)\b\w+\b")
    X = vectorizer.fit_transform(kmer_docs).toarray()
    
    # Create a mapping from epitope names to integers
    unique_epitopes = filtered_df[label_col].unique()
    epitope_to_int = {epitope: i for i, epitope in enumerate(unique_epitopes)}
    
    # Transform labels into integers based on the mapping
    y = filtered_df[label_col].map(epitope_to_int).values
    
    # Get unique k-mer names used in the matrix
    feature_names = vectorizer.get_feature_names_out()
    epitope_names = list(epitope_to_int.keys())
    # Return the adjusted outputs
    return X, y, feature_names, kmer_count_dict, epitope_names

def _cal_micro_ROC(y_test, y_score):
    """Calculate the micro ROC value"""
    fpr, tpr, _ = roc_curve(y_test.ravel(), y_score.ravel())
    return fpr, tpr, auc(fpr, tpr)


def _cal_macro_ROC(y_test, y_score, fpr, tpr, n_classes):
    """Calculate the macro ROC value"""
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes
    return all_fpr, mean_tpr, auc(all_fpr, mean_tpr)


def _plot_roc_curves(fpr, tpr, roc_auc, epi_list, title):
    """PLot the ROC curve"""
    mean_fpr = np.linspace(0, 1, 200)
    tprs = list()
    aucs = list()
    for i in range(len(epi_list)):
        tprs.append(interp(mean_fpr, fpr[i], tpr[i]))
        tprs[-1][0] = 0.0
        aucs.append(roc_auc[i])
        cur_auc = round(roc_auc[i], 3)
        plt.plot(fpr[i], tpr[i], lw=1, alpha=0.5, label='{0}({1})'.format(epi_list[i], str(cur_auc)))

    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(tprs, axis=0)
    plt.plot(mean_fpr, mean_tpr, color='b', label='Mean ROC({})'.format(round(mean_auc, 3)), lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='gray', alpha=.2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc='lower right')
    plt.show()


def _cal_roc_auc(y_test, y_score, y_pred, epi_list, draw_roc_curve=True, title="ROC curves"):
    """"Calculate the AUROC value and draw the ROC curve."""
    fpr = dict()
    tpr = dict()
    precision = list()
    recall = list()
    roc_auc = dict()
    y_test = label_binarize(y_test, classes=np.arange(len(epi_list)))
    y_pred = label_binarize(y_pred, classes=np.arange(len(epi_list)))
    for i in range(len(epi_list)):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        precision.append(precision_score(y_test[:, i], y_pred[:, i]))
        recall.append(recall_score(y_test[:, i], y_pred[:, i]))

    # micro-average ROC
    fpr["micro"], tpr["micro"], roc_auc["micro"] = _cal_micro_ROC(y_test, y_score)

    # macro-average ROC
    fpr["macro"], tpr["macro"], roc_auc["macro"] = _cal_macro_ROC(y_test, y_score, fpr, tpr, len(epi_list))

    # plot all ROC curves
    if draw_roc_curve:
        _plot_roc_curves(fpr, tpr, roc_auc, epi_list, title)

    return roc_auc, np.mean(precision), np.mean(recall)


def predict_auc(X, y, classifier, cv, epi_list, draw_roc_curve=True, title="ROC curves"):
    auc_dict = {}
    acc_list, precision_list, recall_list = [], [], []
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=666)
    cur_fold = 1
    for train_index, test_index in skf.split(X, y):
        # split cross-validation folds
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train, X_test = pca_analyse(X_train, X_test, 0.9)

        clf = clone(classifier)
        clf.fit(X_train, y_train)

        acc_list.append(clf.score(X_test, y_test))

        y_prob = clf.predict_proba(X_test)
        y_pred = clf.predict(X_test)

        auc_dict[cur_fold], precision, recall = _cal_roc_auc(y_test, y_prob, y_pred, epi_list, draw_roc_curve)

        precision_list.append(precision)
        recall_list.append(recall)
        cur_fold += 1

    return auc_dict, acc_list, precision_list, recall_list


def pca_analyse(X_train, X_test, rate=0.9):
    """Perform PCA for the train set and test set."""
    pca = PCA(n_components=rate).fit(X_train)
    return pca.transform(X_train), pca.transform(X_test)

