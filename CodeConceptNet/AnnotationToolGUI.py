import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont
import json  # Import the json module to work with JSON files

# import numpy as np
# from PIL import Image
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# from nltk.corpus import stopwords

cluster_id = None
cluster_id = None
current_cluster_index = 0
clusters_data = None


def update_json_with_user_input(cluster_id, meaningful,lex_input, syn_input, sem_input):
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        if cluster_id in data:
            print("Updating JSON with input: ", meaningful, lex_input, syn_input, sem_input)
            data[cluster_id][-1]["Meaningful"] = meaningful
            if lex_input:
                data[cluster_id][-1]["Lexicographic"] = lex_input
            if syn_input:
                data[cluster_id][-1]["Syntactic"] = syn_input
            if sem_input:
                data[cluster_id][-1]["Semantic"] = sem_input

        else:
            print(f"Error writing to JSON: Cluster ID {cluster_id} not found")

        with open(json_file_path, 'w') as file:
            json.dump(data, file, indent=4)
    except IOError as e:
        print("An error occurred while writing to the JSON file:", e)

def load_next_cluster_data():
    global current_cluster_index, clusters_data
    cluster_ids = list(clusters_data.keys())

    while current_cluster_index < len(cluster_ids):
        current_cluster = cluster_ids[current_cluster_index]
        # Check if the current cluster data has a 'UserInput' key
        if "Meaningful" not in clusters_data[current_cluster][-1]:  # Assuming 'UserInput' would be in the last dict
            load_cluster_data(current_cluster)
            break
        else:
            current_cluster_index += 1  # Move to the next cluster if 'UserInput' exists

    if current_cluster_index >= len(cluster_ids):
        print("No more clusters to display or all remaining clusters have user input.")


def load_cluster_data(cluster_id):
    global current_cluster_index, clusters_data
    print("Current cluster index: ", current_cluster_index)
    # with (open(sentences_file_path, "r")) as sentencesFile:
    #     sentences = sentencesFile.readlines()

    with (open(labels_file_path, "r")) as labelsFile:
        label_lines = [line.strip().split() for line in labelsFile]

    # Clear the Treeview and Labels Text widget
    treeview.delete(*treeview.get_children())
    labels_text.configure(state="normal")
    labels_text.delete("1.0", tk.END)

    cluster_data = clusters_data[cluster_id]

    labels = cluster_data[-1]['Labels']
    allTokens = cluster_data[:-1]


    for entry in allTokens:
        token = entry["Word"]
        sentence_id = int(entry["SentID"])
        token_id = int(entry["TokenID"])
        sentence = entry["Context"]
        try:
            token_label = label_lines[sentence_id][token_id]
        except IndexError:
            token_label = "N/A"


        treeview.insert("", tk.END, values=(token, token_label, sentence))

    for label in labels[:3]:
        labels_text.insert(tk.END, label + "\n\n")
    labels_text.configure(state="disabled")

def on_enter_click():
    global current_cluster_index
    current_cluster = list(clusters_data.keys())[current_cluster_index]
    lex_text = lex_input.get()
    syn_text = syn_input.get()
    sem_text = sem_input.get()
    meaningful = meaningful_answer.get()
    update_json_with_user_input(current_cluster, meaningful, lex_text, syn_text, sem_text)
    lex_input.delete(0, tk.END)
    syn_input.delete(0, tk.END)
    sem_input.delete(0, tk.END)
    current_cluster_index += 1
    load_next_cluster_data()


root = tk.Tk()
root.title("Labelling Tool")




# Frame for the new textboxes
questions_frame = tk.Frame(root)
questions_frame.pack(fill=tk.X)  # Ensure this frame is packed before the top_frame

# Lexicographic? Label and Textbox
lex_label = tk.Label(questions_frame, text="Lexicographic?")
lex_label.pack(side=tk.LEFT, padx=(10, 2), pady=10)
lex_input = tk.Entry(questions_frame)
lex_input.pack(side=tk.LEFT, expand=False, fill=tk.X, padx=(0, 10), pady=10)

# Syntactic? Label and Textbox
syn_label = tk.Label(questions_frame, text="Syntactic?")
syn_label.pack(side=tk.LEFT, padx=(10, 2), pady=10)
syn_input = tk.Entry(questions_frame)
syn_input.pack(side=tk.LEFT, expand=False, fill=tk.X, padx=(0, 10), pady=10)

# Semantic? Label and Textbox
sem_label = tk.Label(questions_frame, text="Semantic?")
sem_label.pack(side=tk.LEFT, padx=(10, 2), pady=10)
sem_input = tk.Entry(questions_frame)
sem_input.pack(side=tk.LEFT, expand=False, fill=tk.X, padx=(0, 10), pady=10)

# Frame for the meaningful question
meaningful_frame = tk.Frame(root)
meaningful_frame.pack(fill=tk.X, before=questions_frame)  # Ensure this frame is packed before the questions_frame


# Enter button next to the input
enter_button = tk.Button(questions_frame, text="Enter")
enter_button.pack(side=tk.LEFT, padx=(10, 0), pady=10)

# Label for the question
meaningful_label = tk.Label(meaningful_frame, text="Is the cluster meaningful?")
meaningful_label.pack(side=tk.LEFT, padx=(10, 2), pady=10)

# Variable to hold the answer
meaningful_answer = tk.StringVar(value="I don't know")  # Default value

# Radio buttons for the answers
yes_rb = tk.Radiobutton(meaningful_frame, text="Yes", variable=meaningful_answer, value="Yes")
yes_rb.pack(side=tk.LEFT, padx=(10, 2), pady=10)

no_rb = tk.Radiobutton(meaningful_frame, text="No", variable=meaningful_answer, value="No")
no_rb.pack(side=tk.LEFT, padx=(10, 2), pady=10)

idk_rb = tk.Radiobutton(meaningful_frame, text="I don't know", variable=meaningful_answer, value="I don't know")
idk_rb.pack(side=tk.LEFT, padx=(10, 2), pady=10)



# Frame for displaying labels, placed below the user input and above the Treeview
labels_frame = tk.Frame(root, height=100)  # Adjust height as needed
labels_frame.pack(fill=tk.X, pady=10)



# Title label for the LLM Suggestions Text widget
llm_title_label = tk.Label(labels_frame, text="LLM Suggestions", font=("Arial", 12, "bold"))
llm_title_label.pack(side=tk.TOP, fill=tk.X)

# Text widget for displaying labels
labels_text = tk.Text(labels_frame, height=4, wrap="word")
labels_text.pack(side=tk.LEFT, fill=tk.X, expand=True)
labels_text.configure(state="disabled")  # Start as read-only

# Scrollbar for the Text widget
labels_scroll = ttk.Scrollbar(labels_frame, orient="vertical", command=labels_text.yview)
labels_scroll.pack(side=tk.RIGHT, fill="y")
labels_text.configure(yscrollcommand=labels_scroll.set)

# Create a frame for the Treeview widget to allow for more flexible resizing
frame = tk.Frame(root)
frame.pack(fill=tk.BOTH, expand=True)

# Define the Treeview widget with the desired columns
treeview = ttk.Treeview(frame, columns=("Cluster # words", "Token Label", "Sentence Context"), show="headings")
treeview.heading("Cluster # words", text="Words from Cluster #")
treeview.heading("Token Label", text="Token's Label")
treeview.heading("Sentence Context", text="Context from Sentence")
treeview.column("Cluster # words", stretch=tk.YES, width=10)
treeview.column("Token Label", stretch=tk.YES, width=10)
treeview.column("Sentence Context", stretch=tk.YES, width=300)
treeview.pack(fill=tk.BOTH, expand=True)

customFont = tkFont.Font(family="Helvetica", size=12)  # Adjust the size as needed
style = ttk.Style()
style.configure("Treeview", font=customFont, rowheight=customFont.metrics("linespace"))


enter_button.config(command=on_enter_click)


json_file_path = "451-500.json"
labels_file_path = "codetest2_test_unique.label"

with open(json_file_path, "r") as jsonFile:
    clusters_data = json.load(jsonFile)

load_next_cluster_data();


root.mainloop()
