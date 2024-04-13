import os
import argparse
import numpy as np
import time
from aux_classifier import utils

def main():
    #configurations = [
    #    ("Total", 1000000, 9984),
    #    ("BERT", 1000000, 4608),
    #    ("CodeBERT", 1000000, 1536),
    #    ("GraphCodeBERT", 1000000, 768),
    #    ("UniXCoder", 1000000, 768),
    #    ("CGJ", 1000000, 9984),
    #    ("CGP", 1000000, 768),
    #    ("RoBERTa", 1000000, 9984)
    #]

    configurations = [
        ("Total", 1000000, 9984),
        ("BERT", 1000000, 299),
        ("CodeBERT", 1000000, 299),
        ("GraphCodeBERT", 1000000, 599),
        ("UniXCoder", 1000000, 599),
        ("CGJ", 1000000, 299),
        ("CGP", 1000000, 49),
        ("RoBERTa", 1000000, 299)
    ]

    results = []  # List to store results

    for model, num_instances, num_features in configurations:
        X = np.random.random((num_instances, num_features)).astype(np.float32)
        y = np.random.randint(0, 2, (num_instances,))  # Assuming binary classification

        start_time = time.process_time()
        print("Global start time:", start_time)
        trained_model = utils.train_logreg_model(
            X,
            y,
            lambda_l1=0.00001,
            lambda_l2=0.00001,
            num_epochs=10,
            batch_size=128
        )
        end_time = time.process_time()
        print("Global end time:", end_time)
        result = "Total time: %0.4f seconds (Model: %s, Num Instances: %d, Num Features: %d)" % (end_time - start_time, model, num_instances, num_features)
        print(result)
        results.append(result)

    # Print all results together
    print("\n".join(results))
    print("Token Tagging")

if __name__ == "__main__":
    main()

