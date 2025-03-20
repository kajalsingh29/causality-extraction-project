import spacy
import json
from dataprocess_utils import *
from get_embeddings_function import *
from langchain.vectorstores.chroma import Chroma
from langchain_community.llms.ollama import Ollama
from graph_creation import *
from relations import relations_chemprot
import re
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from tqdm import tqdm
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
CHROMA_PATH = "chroma_chemprot"


def read_json(file_path):
    """
    Reads a JSON file and returns its content as a string.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        str: The JSON content as a string.
    """
    try:
        with open(file_path, 'r') as file:
            json_content = json.load(file)
        # Convert JSON object to a string
        json_string = json.dumps(json_content, indent=2)
        return json_string
    except FileNotFoundError:
        return f"File not found: {file_path}"
    except json.JSONDecodeError:
        return f"Error decoding JSON from file: {file_path}"
    
  
def load_json(json_string):
    json_pattern = r'\{.*\}'
    match = re.search(json_pattern, json_string, re.DOTALL)

    if match:
        json_data = match.group(0)
        # Convert to a valid JSON string

        # Print the extracted JSON

        # Optionally, load the JSON to verify it's valid
        
        try:
            json_object = json.loads(json_data)
            print("Extracted JSON is valid.")
            return json_object
        except json.JSONDecodeError as e:
            print("Invalid JSON:", e)
            return None
        
    else:
        print("No JSON found in the input text.")

train_path = 'C:/Users/USER/Desktop/final_project/Chemprot/train.txt'
test_path = 'C:/Users/USER/Desktop/final_project/Chemprot/test.txt'
val_path = 'C:/Users/USER/Desktop/final_project/Chemprot/dev.txt'

# Load the JSON files
with open(train_path, 'r', encoding='utf-8') as file:
        train_data_j = file.read()
        # train_data_j = json.load(file)

with open(test_path, 'r', encoding='utf-8') as file:
    test_data_j = file.read()
    # test_data_j = train_data_j.split('\n')

with open(val_path, 'r', encoding='utf-8') as file:
    val_data_j = file.read()
    # val_data_j = train_data_j.split('\n')


train_data_j = train_data_j.split('\n')
train_data_j = train_data_j[:-1]
test_data_j = test_data_j.split('\n')
test_data_j = test_data_j[:-1]
val_data_j = val_data_j.split('\n')
val_data_j = val_data_j[:-1]

# Extract documents information
all_train_documents_information = train_data_j
all_test_documents_information = test_data_j
all_val_documents_information = val_data_j

# Create readable chunks (assuming the function is defined in dataprocess_utils)
_, test_data, _ = creating_readable_chunks_for_chemprot(all_train_documents_information, all_test_documents_information, all_val_documents_information)
response_file_path = "model_responses.txt"
error_path = "error.txt"
# random.shuffle(test_data)
model = Ollama(model="llama3")

true_labels = []
predicted_labels = []

random.seed(42)

random.shuffle(test_data)


print(len(test_data))
def process_test_data(i):
    text = test_data[i]["text"]
    correct_label = test_data[i]['label']
    list_of_entities = [f"{ent}" for ent in test_data[i]["entities"]]

    # Get embedding function and search with Chroma
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    search_text = test_data[i]['text'] + f'{list_of_entities}'
    results = db.similarity_search_with_score(test_data[i]['text'], k=5)

    # Example string formatting
    EXAMPLES = """\nExample: {Number} \nSentence: {sentence} \nEntities: {entities} \nRelations: {relations}"""
    example = ""
    cnt = 1
    for document, score in results:
        EXAMPLES = EXAMPLES.format(
            Number=cnt, 
            sentence=document.page_content, 
            entities=document.metadata['entities'], 
            relations=document.metadata['relations']
        )
        example += EXAMPLES
        EXAMPLES = """
        Example: {Number}
        Text: {sentence}
        Entities: {entities}
        Relations: {relations}
        """
        cnt += 1

    # Create the prompt
    prompt = f"""
     You are an expert in extracting relationships from sentences containing biomedical data. Your task is to identify the relation between the entities provided to you by taking reference from the sentence.

    ##Instructions:

    You have to find the relation between the entities mentioned here: 
    {list_of_entities}

    Here is the sentence: 
    {test_data[i]['text']}

    These are the relation types you have to choose from:
    {relations_chemprot}

    ## Examples:
    Here is the relevant context and examples of how to extract relations from a sentence:
    {example}

    Please provide your answer in the following JSON format. Include an explanation of why entity1 is related to entity2 for your better understanding:
    {read_json("output_template.json")}

    Note that the sentence might not provide information about a direct relationship between entities. In such cases, you have to infer a relation between them by using information from examples.
    """
 
    # Invoke the model
    response_text = model.invoke(prompt)
    
    # Save the response_text to a file
    with open(response_file_path, 'a', encoding='utf-8') as file:
        print(f"Response received for test data {i}")
        file.write(f"Response for test data {i}:\n")
        file.write(response_text)
        file.write("\n\n")
    
    # Process the response and return true and predicted labels
    try:
        output_json = load_json(response_text)
        if output_json and output_json["Relation"]:
            predicted_label = output_json["Relation"] 
        else:
            output_json = None
            predicted_label = None
        file_path = 'output.txt'
# Open the file in write mode
        with open(file_path, 'a', encoding='utf-8') as file:
            # Iterate over the list and write each item to the file
            file.write(f"{predicted_label},{correct_label}\n")  # Writing each item on a new line
        return correct_label, predicted_label
    except Exception:
        print(f"Error with test data {i}, skipping this instance.")
        with open(error_path, 'a', encoding='utf-8') as file:
            file.write(f"Error with test data {i}, skipping this instance.\n")
            file.write(response_text)
            file.write("\n\n")
        return None, None  # Return None for skipped instances
import os
# Use ThreadPoolExecutor to parallelize the processing
with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    # Wrap with tqdm to track progress
    futures = [executor.submit(process_test_data, i) for i in range(len(test_data))]
    results = [future.result() for future in as_completed(futures)]
    # Track progress as tasks complete
    for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing test data"):
        pass

print(results)

# Collect the true and predicted labels
for correct_label, predicted_label in results:
    if correct_label is not None and predicted_label is not None:
        true_labels.append(correct_label)
        predicted_labels.append(predicted_label)

# Calculate the F1 score
f1 = f1_score(true_labels, predicted_labels, average='micro')
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='micro')
recall = recall_score(true_labels, predicted_labels, average='micro')

f1, accuracy, precision, recall
print(f"F1 Score: {f1}")
print(f"accuracy: {accuracy}")
print(f"precision: {precision}")
print(f"recall: {recall}")
