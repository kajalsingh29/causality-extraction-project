import pandas as pd
import os
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
from concurrent.futures import ThreadPoolExecutor, as_completed

test_path = 'C:/Users/USER/Desktop/final_project/Chemprot/test.csv'

CHROMA_PATH = "chroma_chemprot"
response_file_path = "model_responses.txt"
error_path = "error.txt"
output_file_path = "output.txt"
test_data_file = test_path

# Load test_data.csv into a list of dictionaries
def load_test_data(file_path):
    df = pd.read_csv(file_path)
    # Convert DataFrame rows to a list of dictionaries
    return df.to_dict(orient='records')

test_data = load_test_data(test_data_file)

# Prepare the model and variables

true_labels = []
predicted_labels = []

# Ensure reproducibility



print(f"Number of test samples: {len(test_data)}")

def read_json(file_path):
    """
    Reads a JSON file and returns its content as a string.
    """
    try:
        with open(file_path, 'r') as file:
            json_content = json.load(file)
        json_string = json.dumps(json_content, indent=2)
        return json_string
    except FileNotFoundError:
        return f"File not found: {file_path}"
    except json.JSONDecodeError:
        return f"Error decoding JSON from file: {file_path}"

def load_json(json_string):
    """
    Extract JSON data from a response string.
    """
    json_pattern = r'\{.*\}'
    match = re.search(json_pattern, json_string, re.DOTALL)
    if match:
        json_data = match.group(0)
        try:
            json_object = json.loads(json_data)
            return json_object
        except json.JSONDecodeError:
            return None
    return None

SYSTEM = ""
model = Ollama(model="llama3relations", cache = False, system=SYSTEM, top_k = 10, top_p = 0.3)
def process_test_data(i):
    """
    Process a single test data instance and predict the relation.
    """
    
    text = test_data[i]["text"]
    correct_label = test_data[i]["label"]
    list_of_entities = test_data[i]["entities"]

    # Get embedding function and search with Chroma
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    search_text = test_data[i]['text'] + f'{list_of_entities}'
    results = db.similarity_search_with_score(test_data[i]['text'], k=5)

    # Prepare examples for the prompt
    EXAMPLES = """\nExample: {Number} \nText: {sentence} \nEntities: {entities} \nRelations: {relations}"""
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
    prompt = f'''
    """
    Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    Analyse the entities mentioned in the Input sentence and determine the relation between these entities. You can select one of these relations: 'PRODUCT-OF', 'AGONIST-ACTIVATOR', 'INHIBITOR', 'ANTAGONIST', 'ACTIVATOR', 'AGONIST-INHIBITOR', 'DOWNREGULATOR', 'AGONIST', 'SUBSTRATE_PRODUCT-OF', 'UPREGULATOR', 'INDIRECT-UPREGULATOR', 'INDIRECT-DOWNREGULATOR', 'SUBSTRATE'

    ### Input:
    Sentence: {test_data[i]['text']}
    Entities:  {list_of_entities}

    ### Response:

    """
    '''
    print(prompt)
    # Invoke the model
    response_text = model.invoke(prompt)
    print(response_text)
    predicted_label = response_text.strip()
    with open(output_file_path, 'a', encoding='utf-8') as file:
        file.write(f"{predicted_label},{correct_label}\n")
    return correct_label, predicted_label
    # Save the response_text to a file
    # with open(response_file_path, 'a', encoding='utf-8') as file:
    #     file.write(f"Response for test data {i}:\n{response_text}\n\n")
    
    # # Process the response and return true and predicted labels
    # try:
    #     output_json = load_json(response_text)
    #     predicted_label = output_json.get("Relation") if output_json else None
    #     with open(output_file_path, 'a', encoding='utf-8') as file:
    #         file.write(f"{predicted_label},{correct_label}\n")
    #     return correct_label, predicted_label
    # except Exception:
    #     with open(error_path, 'a', encoding='utf-8') as file:
    #         file.write(f"Error with test data {i}, skipping this instance.\n{response_text}\n\n")
    #     return None, None


# with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
#     futures = [executor.submit(process_test_data, i) for i in range(len(test_data))]
#     results = [future.result() for future in as_completed(futures)]
#     for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing test data"):
#         pass



results = []
# Collect true and predicted labels
for i in range(len(test_data)):
    ans = process_test_data(i)
    results.append(ans)

for correct_label, predicted_label in results:
    if correct_label is not None and predicted_label is not None:
        true_labels.append(correct_label)
        predicted_labels.append(predicted_label)


# Calculate evaluation metrics
f1 = f1_score(true_labels, predicted_labels, average='micro')
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='micro')
recall = recall_score(true_labels, predicted_labels, average='micro')

# Print metrics
print(f"F1 Score: {f1}")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
