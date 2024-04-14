# Discovering Latent Concepts in CodeBERT

This project is a replication of ConceptX. For the original implementation and more details, visit the [ConceptX repository](https://github.com/hsajjad/ConceptX) on GitHub.

We include the original clusters generated using ConceptX in 

## Annotation Tool
The code for the annotation tool is provided in `AnnotationToolGUI.py`

![image](https://github.com/arushisharma17/Redundancy-and-Concept-Analysis-for-Code-Trained-Language-Models/assets/28835447/763d90f9-afea-41c1-9462-2d4dd7e32fa8)

## ConceptNetDataset

We provide the labelled dataset in `CodeConceptNet.json`, which comprises a JSON file with 500 labelled clusters. The format is organized by cluster IDs, each containing a list of JSON objects that describe the usage of various code elements. Below is a detailed structure of what each cluster entry typically includes:

### Cluster ID Format
Each cluster ID (e.g., `289`) represents a specific concept or usage pattern in code. Under each cluster ID, there are several entries, each with multiple attributes:

- **Word**: The character or string being analyzed (e.g., `;`).
- **WordID**: A unique identifier for each occurrence of the word.
- **SentID**: The sentence or statement ID where the word appears.
- **TokenID**: The token ID within the sentence or statement.
- **Context**: The actual code snippet illustrating the use of the word.

In addition to these attributes, the cluster itself may include additional metadata at the same level as the cluster ID:

- **Labels**: Descriptions or tags associated with the usage pattern of the cluster (e.g., `"End of Statement / Line Terminator"`).
- **UserInput**: A summary provided by users, detailing the relevance and thematic classification of the cluster (e.g., "Meaningful: Yes. Theme: Syntax Character. This cluster identifies semicolons which are statement terminators").
- **Meaningful**: Indicates whether the occurrence is meaningful within the context.
- **Syntactic**: Descriptive label of the syntactic function (e.g., "Semicolon").
