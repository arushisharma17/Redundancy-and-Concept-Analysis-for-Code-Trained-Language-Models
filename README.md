# Redundancy and Concept Analysis for Code-trained Language Models
This repository contains the code for the paper Redundancy and Concept Analysis for Code-trained Language Models

## Part 1: Redundancy Analysis for Downstream Tasks (RQ 1 and RQ2)

### Getting the Data

- **Defect Detection**: [https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection)
- **Clone Detection**: [https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Clone-detection-BigCloneBench](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Clone-detection-BigCloneBench)
- **NL-Code Search**: [https://github.com/microsoft/CodeXGLUE/tree/main/Text-Code/NL-code-search-WebQuery](https://github.com/microsoft/CodeXGLUE/tree/main/Text-Code/NL-code-search-WebQuery)


### Getting neuron activations

Below is a summary of the dataset statistics for neuron activations, showcasing the distribution across training, development, and test sets, along with the number of tags for each task.

| Task                         | Train | Dev  | Test  | Tags |
|------------------------------|-------|------|-------|------|
| Token-tagging (filtered)     | 12000 | 1620 | 2040  | 6    |
| Defect detection [Zhou et al., 2019](https://doi.org/10.1109/ICSE.2019.00132) | 19668 | 2186 | 2732  | 2    |
| Clone Detection [Svajlenko et al., 2015](https://doi.org/10.1109/ICSME.2015.7332475) | 81029 | 9003 | 10766 | 2    |
| Code Search [Huang et al., 2020](https://doi.org/10.1145/3397481.3450678) | 18000 | 2000 | 604   | 2    |



### Getting activations

Create the environment: 
`pip install -r finetuning_requirements.txt`

`cd ./Redundancy Analysis`

Each task directory contains the code for two steps:
1. Finetuning 
2. Extraction 

cd into the task directory to get specific instructions for that task.

### Performing experiments

Create the environment: 
`pip install -r experiments_requirements.txt`

There are three main helper scripts provided: 

1. `experiments/classification/run_sentence_pipeline_all.py`
Produces oracle numbers, individual classifier layer numbers, and concatenated classifier numbers. Additionally, given a list of correlation clustering coefficients and performance deltas for LayerSelector and CCFS, this script calculates the corresponding accuracies. 

2. `experiments/classification/run_sentence_cc_all.py`
Generates oracle numbers and performance numbers at all correlation clustering thresholds.

3. `experiments/classification/run_sentence_max_features.py`
   Produces oracle numbers and identifies the minimal set of neurons from all neurons for accuracies.

## Part 2: Redundancy and Concept Analysis for Token Tagging Task (RQ3)

## Getting the data
1. Raw code: Deduplicated Java code in [https://github.com/dashends/CodeSyntax](https://github.com/dashends/CodeSyntax).
2. Preprocess the code and generate labels: Run `./Rundancy Analysis/Token Tagging/Preprocessing.py`
The processed tokens and labels are in the folder: `./Rundancy Analysis/Token Tagging/src_java`

### Performing experiments
Run `run_neurox1.py` in folder `./Rundancy Analysis/Token Tagging/` with the param `extract=False` if you have already
had the activation files. If not, make `extract=True`

You may want to save the output to a log file for future analysis.

### Get the important neurons to properties of code:
Examine the log file of the token tagging task, the important neurons to properties of code can be found 
based on the top words of neurons.

### Get the activated words of neurons:
After examining the top words, you have the indices and the words, then run the script 
`./Rundancy Analysis/Token Tagging/TracingConcept.py` with the token and neuron indices.

### CKA
For all the tasks, run `./Rundancy Analysis/Token Tagging/CKA/cka.py`


## Part 3: CodeConceptNet Dataset (RQ4)

