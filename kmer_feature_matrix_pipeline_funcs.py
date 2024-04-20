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
from sklearn.metrics import classification_report, confusion_matrix

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
    return X, y, feature_names, kmer_count_dict, epitope_names, epitope_to_int

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

def _plot_roc_curves_mean_only(fpr, tpr, roc_auc, epi_list, title):
    """Plot only the mean ROC curve."""
    mean_fpr = np.linspace(0, 1, 200)
    tprs = []
    aucs = []

    # Calculate mean and standard deviation of TPRs for all epitopes
    for i in range(len(epi_list)):
        tprs.append(np.interp(mean_fpr, fpr[i], tpr[i]))
        tprs[-1][0] = 0.0
        aucs.append(roc_auc[i])

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_tpr = np.std(tprs, axis=0)

    # Plotting only the mean ROC curve
    plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {round(mean_auc, 3)})', lw=2, alpha=0.8)
    plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color='gray', alpha=0.2)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.5)
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
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
        _plot_roc_curves_mean_only(fpr, tpr, roc_auc, epi_list, title)

    return roc_auc, np.mean(precision), np.mean(recall)


def predict_auc(X, y, classifier, cv, epi_list, draw_roc_curve=True, title="ROC curves"):
    auc_dict = {}
    acc_list, precision_list, recall_list = [], [], []
    all_conf_matrices = []
    all_class_reports = []
    misclassified_instances = []
    misclassified_details = []
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=666)
    cur_fold = 1
    for train_index, test_index in skf.split(X, y):
        # split cross-validation folds
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #X_train, X_test = pca_analyse(X_train, X_test, 0.9)

        clf = clone(classifier)
        clf.fit(X_train, y_train)

        acc_list.append(clf.score(X_test, y_test))

        y_prob = clf.predict_proba(X_test)
        y_pred = clf.predict(X_test)

        for idx, (true_label, pred_label) in enumerate(zip(y_test, y_pred)):
            if true_label != pred_label:
                misclassified_details.append((test_index[idx], true_label, pred_label))

        # Save classification report and confusion matrix
        class_report = classification_report(y_test, y_pred, target_names=epi_list, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        all_class_reports.append(class_report)
        all_conf_matrices.append(conf_matrix)

        auc_dict[cur_fold], precision, recall = _cal_roc_auc(y_test, y_prob, y_pred, epi_list, draw_roc_curve)

        precision_list.append(precision)
        recall_list.append(recall)
        # Track misclassified instances
        mis_indices = test_index[np.where(y_test != y_pred)]
        misclassified_instances.extend(mis_indices)
 
        cur_fold += 1

    # Optionally, plot the confusion matrix of the last fold
    plt.figure(figsize=(10, 7))
    sns.heatmap(all_conf_matrices[0], annot=True, fmt="d", xticklabels=epi_list, yticklabels=epi_list)
    plt.title("Confusion Matrix")
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.show()

    return auc_dict, acc_list, precision_list, recall_list, all_class_reports, all_conf_matrices, clf, misclassified_instances, misclassified_details


def pca_analyse(X_train, X_test, rate=0.9):
    """Perform PCA for the train set and test set."""
    pca = PCA(n_components=rate).fit(X_train)
    return pca.transform(X_train), pca.transform(X_test)

def plot_feature_importance(classifier, feature_names, top_n=20):
    # Get feature importances from the classifier
    importances = classifier.feature_importances_
    
    # Create a list of tuples (feature_name, importance)
    feature_importance = list(zip(feature_names, importances))
    
    # Sort the feature importances by most important first
    feature_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)
    
    # Taking the top n features
    top_features = feature_importance[:top_n]
    features, scores = zip(*top_features)
    
    # Plotting
    y_pos = np.arange(len(features))
    plt.barh(y_pos, scores, align='center', alpha=0.5)
    plt.yticks(y_pos, features)
    plt.xlabel('Importance')
    plt.title('Top {} Feature Importances'.format(top_n))
    plt.gca().invert_yaxis()  # Invert the Y-axis to show the highest value at the top
    plt.show()



def calculate_class_stats(X, y, feature_names):
    """ Calculate statistics for each feature across all classes for one-vs-all. """
    unique_classes = np.unique(y)
    stats = {}
    for cls in unique_classes:
        class_index = y == cls
        non_class_index = ~class_index
        stats[cls] = {
            'mean': np.mean(X[class_index], axis=0),
            'var': np.var(X[class_index], axis=0) + 1e-6,  # Avoid division by zero
            'mean_non_class': np.mean(X[non_class_index], axis=0),
            'var_non_class': np.var(X[non_class_index], axis=0) + 1e-6
        }
    return stats

def compute_1_DBC_scores(stats):
    """ Compute 1-DBC scores for each feature across all classes. """
    scores = {}
    for cls, data in stats.items():
        score = (data['mean'] - data['mean_non_class']) / (data['var'] + data['var_non_class'])
        scores[cls] = score
    return scores

def select_features(X, feature_names, scores, top_k=50):
    """ Select top features based on 1-DBC scores for all classes and return feature details. """
    all_top_indices = set()
    top_features = {}
    for cls, cls_scores in scores.items():
        top_indices = np.argsort(-np.abs(cls_scores))[:top_k]
        all_top_indices.update(top_indices)
        # Save the top features for this class
        top_features[cls] = {
            'indices': top_indices,
            'names': [feature_names[i] for i in top_indices],
            'scores': cls_scores[top_indices]
        }
    # Filter X for selected indices
    X_selected = X[:, list(all_top_indices)]
    return X_selected, top_features

def plot_top_features(top_features, class_labels):
    """ Plot the top features for each class. """
    for cls in class_labels:
        features = top_features[cls]
        # Create a DataFrame for easier plotting
        df = pd.DataFrame({
            'Feature': features['names'],
            'Score': features['scores']
        })
        df.sort_values(by='Score', ascending=False, inplace=True)
        
        # Plotting
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Score', y='Feature', data=df.head(20))  # Show top 20 features
        plt.title(f'Top Features for Class {cls}')
        plt.xlabel('1-DBC Score')
        plt.ylabel('K-mer')
        plt.show()

""" example usage"""

"""X, y, feature_names, kmer_count_dict, epitope_names, epitope_to_int = create_features_matrix(filtered_df, include_alpha=False, include_beta=True, alpha_col='cdr3.alpha', beta_col='cdr3.beta', label_col='antigen.epitope', k=3)
class_stats = calculate_class_stats(X, y, feature_names)
scores = compute_1_DBC_scores(class_stats)
X_selected, top_features = select_features(X, feature_names, scores, top_k=50)

plot_top_features(top_features, np.unique(y))"""
"""auc_result, acc, precision, recall, class_reports, conf_matrices,clf = predict_auc(X, y, rf_classifier, 2, epitope_names, True)"""
"""plot_feature_importance(clf, feature_names)"""
