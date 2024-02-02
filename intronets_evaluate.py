import h5py
import sys
import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd





def intronets_evaluate_cutoffs(y_true, y_pred, cutoff_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99], plot=False, save_plot=True, save_numbers=False, plot_title="precision recall curve", plot_filename="pr_rec.png", numbers_filename="pr_rec.csv"):
    """
    Description:
        function which computes precision and recall according to the values of the provided cutoff_list

    Arguments:
        y_true np.array: array of true introgression (1: introgressed, 0: not introgressed)
        y_pred np.array: array of predicted introgression (probabilities)
        cutoff_list list: list containing the cutoffs
    """
    
    
    precisions = []
    recalls = []
    for cutoff in cutoff_list:
        all_true_positive = 0
        all_false_positive = 0
        all_false_negative = 0
        for ir, true_row in enumerate(y_true):
            predicted_row = y_pred[ir]


            predicted_labels = [1 if prob >= cutoff else 0 for prob in predicted_row]

            true_positive = sum((true == 1 and pred == 1) for true, pred in zip(true_row, predicted_labels))
            false_positive = sum((true == 0 and pred == 1) for true, pred in zip(true_row, predicted_labels))
            false_negative = sum((true == 1 and pred == 0) for true, pred in zip(true_row, predicted_labels))

            all_true_positive = all_true_positive  + true_positive
            all_false_positive = all_false_positive  + false_positive
            all_false_negative = all_false_negative  + false_negative


        precision = all_true_positive / max((all_true_positive + all_false_positive), 1)
        recall = all_true_positive / max((all_true_positive + all_false_negative), 1)

        precisions.append(precision)
        recalls.append(recall)

    if plot==True:
        plot_cutoffs(recalls, precisions, title=plot_title, plot_filename = plot_filename)


    return precisions, recalls, cutoff_list


def plot_cutoffs(recs, precs, title=None, save=True, plot_filename = "prc_rec.png"):
    '''
    simple plot function for precision-recall
    '''
    plt.plot(recs, precs)
    plt.scatter(recs, precs)
    plt.xlim(left=0)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve for computed cutoffs")
    if title != None:
        plt.title(title)

    if save == True:
        plt.savefig(plot_filename)
    plt.show()



def write_prec_recall_df(cut_offs, precisions, recalls, model_name = "archie", sample_name = "example", acc_filename="pr_rec.csv"):
    """
    Description:
        this function writes precision-recall-information to file

    Arguments:
        cut_offs list: list containing the used cut-ffs
        precisions list: list containing the computed precision values
        recalls int: list containing the computed recall values
        model_name str: name of the model
        sample_name str: name of the sample
        acc_filename str: arbitrary name for output files
    """

    prec_rec_list = []
    for i, cut_off in enumerate(cut_offs):
        prec_rec_list.append([model_name, sample_name, cut_off, precisions[i], recalls[i]])

    prec_rec_df = pd.DataFrame(prec_rec_list)
    prec_rec_df.columns = ["demography","sample","cutoff","precision","recall"]

    if not os.path.exists(os.path.join("results", "inference", "intronets")):
            os.makedirs(os.path.join("results", "inference", "intronets"))

    prec_rec_df.to_csv(os.path.join("results", "inference", "intronets",acc_filename), sep="\t", index=False, na_rep="nan")



