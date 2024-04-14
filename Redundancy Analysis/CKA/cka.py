import argparse
import neurox.data.loader as data_loader
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1
# 
# ref: https://github.com/AntixK/PyTorch-Model-Compare
# for defect detection, among the 768 dimensions for RoBERTa and UniXCoder models by the embedding layer, a lot of samples have the
# same values which lead to negative values in the sqrt function.
MODEL_NAMES = ['BERT','CodeBERT','GraphCodeBERT','CodeGPTJava','CodeGPTPy','RoBERTa','UniXCoder']
# MODEL_NAMES = ['BERT','CodeBERT']
# MODEL_NAMES = ['GraphCodeBERT','CodeGPTJava','CodeGPTPy','RoBERTa','UniXCoder']

ACTIVATION_NAMES = {'BERT':'bert_activations_train.json',
                    'CodeBERT':'codebert_activations_train.json',
                    'GraphCodeBERT':'graphcodebert_activations_train.json',
                    'CodeGPTJava':'codeGPTJava_activations_train.json',
                    'CodeGPTPy':'codeGPTPy_activations_train.json',
                    'RoBERTa':'RoBERTa_activations_train.json',
                    'UniXCoder':'UniXCoder_activations_train.json'}
ACTIVATION_NAMES_sentence_level = {'BERT':'bert/train_activations.json',
                                  'CodeBERT':'codebert/train_activations.json',
                                  'GraphCodeBERT':'graphcodebert/train_activations.json',
                                  'CodeGPTJava':'codegpt/java-original/train_activations.json',
                                  'CodeGPTPy':'codegpt/python-original/train_activations.json',
                                  'RoBERTa':'roberta/train_activations.json',
                                  'UniXCoder':'unixcoder/train_activations.json'}


N_LAYERs = 13
N_NEUROSN_PER_LAYER = 768
N_SAMPLES = 5000
N_BATCHES = 5

def mkdir_if_needed(dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)


def load_extracted_activations(activation_file_name,activation_folder):
    #Load activations from json files
    activations, num_layers = data_loader.load_activations(f"{activation_folder}/{activation_file_name}")
    return activations


def HSIC(K, L):
        """
        Computes the unbiased estimate of HSIC metric.

        Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
        """
        N = K.shape[0]
        ones = np.ones((N, 1))
        result = np.trace(K @ L)
        result += (ones.transpose() @ K @ ones @ ones.transpose() @ L @ ones) / ((N - 1) * (N - 2))
        result -= (ones.transpose() @ K @ L @ ones) * 2 / (N - 2)
        return 1 / (N * (N - 3)) * result[0,0]


def add_colorbar(im, aspect=10, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    cax.tick_params(labelsize=40)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


def plot_results(hsic_matrix,save_path,title,model_name,task):
        fig, ax = plt.subplots()
        im = ax.imshow(hsic_matrix, origin='lower', cmap='magma')
        if (model_name in ['RoBERTa','UniXCoder'] and task=='DefectDetection') or (task=='HPC' and model_name in ['CodeBERT','GraphCodeBERT']) or (task=="datarace" and model_name in ['GraphCodeBERT','RoBERTa']):
            ax.set_xticks(ticks=[0,5,11])
            ax.set_xticklabels([1,6,12],fontsize=40)

            ax.set_yticks(ticks=[0,5,11])
            ax.set_yticklabels([1,6,12],fontsize=40)
        else:
            ax.set_xticks(ticks=[0,6,12])
            ax.set_xticklabels([0,6,12],fontsize=40)

            ax.set_yticks(ticks=[0,6,12])
            ax.set_yticklabels([0,6,12],fontsize=40)

        ax.set_title(f"{title}", fontsize=50)

        add_colorbar(im)
        plt.tight_layout()

        plt.savefig(save_path, dpi=300)

        plt.show()

def normalize(matrix):
    var_mean = np.mean(matrix,axis=0)
    var_std = np.std(matrix,axis=0)
    return (matrix-var_mean)/(var_std+1e-16)


def cka(activation1,n_samples,model_name,task):
    hsic_matrix = np.zeros((N_LAYERs, N_LAYERs, 3))
    X = np.array([this_token for this_sample in activation1 for this_token in this_sample])
    print(f"Total number of tokens:{X.shape[0]}")
    del activation1
    
    np.random.seed(2)
    num_batches = N_BATCHES
    for this_batch in range(num_batches):
        if task == 'datarace':
            n_samples = len(X)
        random_choice = np.random.choice(len(X),size=n_samples,replace=False)
        random_choice = sorted(random_choice)
        this_sample = X[random_choice]
        # this_sample=normalize(this_sample)
        
        if (model_name in ['RoBERTa','UniXCoder'] and task=='DefectDetection') or (task=='HPC' and model_name in ['CodeBERT','GraphCodeBERT']) or (task=="datarace" and model_name in ['GraphCodeBERT','RoBERTa']):
            starting = 1
        else:
            starting = 0
        for i in range(starting,N_LAYERs):
            index = i*N_NEUROSN_PER_LAYER
            this_X = this_sample[:,index:index+N_NEUROSN_PER_LAYER]
            # The dimension is seq_len X 9984
            K = this_X @ this_X.transpose()
            np.fill_diagonal(K,0.0)

            if (model_name in ['RoBERTa','UniXCoder'] and task=='DefectDetection') or (task=='HPC' and model_name in ['CodeBERT','GraphCodeBERT']) or (task=="datarace" and model_name in ['GraphCodeBERT','RoBERTa']):
                hsic_matrix[i, 1:, 0] += HSIC(K, K) / num_batches
            else:
                hsic_matrix[i, :, 0] += HSIC(K, K) / num_batches
                # print("hsic_matrix[i, :, 0]:")
                # print(hsic_matrix[i, :, 0])


            for j in range(starting,N_LAYERs):
                index = j*N_NEUROSN_PER_LAYER
                this_Y = this_sample[:,index:index+N_NEUROSN_PER_LAYER]
                L = this_Y @ this_Y.transpose()
                np.fill_diagonal(L,0)

                hsic_matrix[i, j, 1] += HSIC(K, L) / num_batches
                hsic_matrix[i, j, 2] += HSIC(L, L) / num_batches
                # print("hsic_matrix[i, j, 2]:")
                # print(hsic_matrix[i, j, 2])

    
    if (model_name in ['RoBERTa','UniXCoder'] and task=='DefectDetection') or (task=='HPC' and model_name in ['CodeBERT','GraphCodeBERT']) or (task=="datarace" and model_name in ['GraphCodeBERT','RoBERTa']):
        dim = np.sqrt(hsic_matrix[1:, 1:, 0]) * np.sqrt(hsic_matrix[1:, 1:, 2])
        hsic_matrix = hsic_matrix[1:, 1:, 1] / dim
    else:
        dim = np.sqrt(hsic_matrix[:, :, 0]) * np.sqrt(hsic_matrix[:, :, 2])
        hsic_matrix = hsic_matrix[:, :, 1] / dim
    assert not np.isnan(hsic_matrix).any(), "HSIC computation resulted in NANs"
    return hsic_matrix


    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default='python')
    args = parser.parse_args()
    task = args.task
    if task == 'java':
        activation_folder = ".../activations_java"
    elif task == 'CloneDetection':
        activation_folder = ".../Clone-Detection/dataset/stratified/activations"
    elif task == 'DefectDetection':
        activation_folder = ".../Defect-detection/dataset/activations"
    elif task == 'CodeSearch':
        activation_folder = ".../NL-Code-Search/NL-code-search-WebQuery/data/activations"

    for this_model in MODEL_NAMES:
        print(f"Generate svg files for {this_model}")
        if task in ['java']:
            this_activation_name = ACTIVATION_NAMES[this_model]
        elif task in ['CloneDetection','DefectDetection','CodeSearch','HPC','datarace']:
            this_activation_name = ACTIVATION_NAMES_sentence_level[this_model]
        else:
            assert 1==0,"Task is not understood"
        activations = load_extracted_activations(this_activation_name,activation_folder)
        print(f"Length of {this_model} activations:",len(activations))
        _, num_neurons = activations[0].shape
        for idx in range(len(activations)):
            assert activations[idx].shape[1] == num_neurons
        print(f"The number of neurons for each token in {this_model}:",num_neurons)
        hsic_matrix = cka(activations,N_SAMPLES,this_model,task)
        print(f"hsic_matrix:")
        print(hsic_matrix)
        del activations
        plot_results(hsic_matrix,save_path=f"./result_all/cka_{task}/{this_model}_cka.png",title=this_model,model_name=this_model,task=task)
        print("-----------------------------------------------------------------")

if __name__ == "__main__":
    main()
