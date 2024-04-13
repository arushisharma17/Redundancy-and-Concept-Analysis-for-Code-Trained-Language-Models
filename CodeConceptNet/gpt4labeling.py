import openai
import os
import json
from dotenv import load_dotenv
import time
from collections import OrderedDict
# Load the environment variable
load_dotenv()

# Set your OpenAI API key here
openai.api_key = os.getenv("OPENAI_API_KEY")
print(openai.api_key)

def ask_gpt41(question, model="gpt-4", temp=0.7, max_retries=5):
    retry_wait = 2  # Start with a 2-second wait
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question}
                ],
                temperature=temp,
                max_tokens=100,
                top_p=1,
                frequency_penalty=0.5,
                presence_penalty=0
            )
            return response.choices[0].message['content'].strip()
        except openai.error.RateLimitError as e:
            print(f"Rate limit exceeded, retrying in {retry_wait} seconds...")
            time.sleep(retry_wait)
            retry_wait *= 2  # Exponential back-off
    raise Exception("Max retries exceeded for GPT-4 request.")

def ask_gpt4(question, model="gpt-4", temp=0.7, max_attempts=5):
    """
    Asks a question to GPT-4 and returns the answer.

    Parameters:
    - question: The question string.
    - model: The model to use. Defaults to "gpt-4-turbo".
    - temp: The temperature for randomness. Lower is less random.

    Returns:
    - The answer as a string.
    """
    #attempt = 0
    #while attempt < max_attempts:
     #   try:
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a software engineering assistant."},
            {"role": "user", "content": question}
            ],
            temperature=temp,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0.5,
            presence_penalty=0
        )
    text = response.choices[0].message['content']
    return text
      #  except openai.error.RateLimitError:
      #      print("Rate limit exceeded, retrying...")
       #     time.sleep((attempt + 1) * 60)  # Exponential back-off could be a more sophisticated approach
       #     attempt += 1
    #return "Unable to generate label due to rate limit issues."

def generate_contextual_labels_for_cluster(entries):
    """
    Generates contextual labels for a given cluster by asking GPT-4, using the 'Word' and 'Context' from entries.
    """
    def unique_ordered_words(entries):
        return list(OrderedDict.fromkeys(entries))

    token_summary = ", ".join(unique_ordered_words([entry['Word'] for entry in entries]))

    # Prepare the summary of code tokens
    #token_summary = ", ".join([entry['Word'] for entry in entries])
    
    # Prepare the summary of context sentences
    context_summary = ". ".join([entry['Context'] for entry in entries if 'Context' in entry][:3])

    # Ensure the context summary ends with a period
    if not context_summary.endswith('.'):
        context_summary += '.'

    # Define the prompts to use for GPT-4 queries
    prompts =[
     #   f"Generate a concise label or theme for the following java code tokens: {token_summary}.",
        f"Given the following java code tokens: {token_summary} and some of their usage contexts: {context_summary},  what functionality or pattern do the tokens represent? Generate a concise label for a cluster of given code tokens.",
 #       f"You are provided a list of java code tokens: {token_summary}, tokenized using an abstract syntax tree tokenizer and clustered using agglomerative hierarchical clustering, along with usage contexts: {context_summary}. Can you label these clusters based on their common traits? The labels should be reflective of the abstract syntax tree and the usage context.",
  #      f"Given the following java code tokens: {token_summary} and the corresponding lines of code which use them: {context_summary}, what functionality or pattern do the tokens represent. Give concise label for a cluster of given code tokens."
    ]

    labels = []
    for prompt in prompts:
        # Ask GPT-4 for each question
        label = ask_gpt4(prompt)
        labels.append(label.strip())

    return labels


def main():
    file_path = "clusters-500.txt"
    json_output_file_path = "grouped_by_cluster-500_2.json"
    sentences_file_path = "/work/LAS/jannesar-lab/arushi/Interpretability/interpretability-of-source-code-transformers/ConceptX/ConceptX/codetest2_test_unique.in"

    # Initialize an empty dictionary to hold the data grouped by ClusterID
    grouped_data = {}

    # Load sentences from the .in file
    with open(sentences_file_path, "r") as sentencesFile:
        sentences = sentencesFile.readlines()

    # Open and read the clusters file
    with open(file_path, 'r') as file:
        for line in file:
            # Split each line by '|||' to get the fields
            word, word_id, sent_id, token_id, cluster_id = line.strip().split('|||')
            # Convert SentID to an integer for indexing
            sent_id = int(sent_id)
            # Fetch the context sentence using SentID
            context_sentence = sentences[sent_id].strip()

            # If the cluster ID is not yet a key in the dictionary, add it with an empty list
            if cluster_id not in grouped_data:
                grouped_data[cluster_id] = []

            # Append a dictionary of the values to the list associated with the cluster ID
            grouped_data[cluster_id].append({
                'Word': word,
                'WordID': word_id,
                'SentID': sent_id,
                'TokenID': token_id,
                'Context': context_sentence  # Add the context sentence
            })
     # Now, generate labels for each cluster
    for cluster_id, entries in grouped_data.items():
        
        # Generate labels for the cluster
        labels = generate_contextual_labels_for_cluster(entries)
        
        # Add the labels to the cluster data
        grouped_data[cluster_id].append({'Labels': labels})

    # Write the grouped data to a JSON file
    with open(json_output_file_path, 'w') as jsonFile:
        json.dump(grouped_data, jsonFile, indent=4)

if __name__ == "__main__":
    main()

