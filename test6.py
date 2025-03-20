train_path = 'C:/Users/USER/Desktop/final_project/Chemprot/train.txt'
test_path = 'C:/Users/USER/Desktop/final_project/Chemprot/test.txt'
val_path = 'C:/Users/USER/Desktop/final_project/Chemprot/dev.txt'
import spacy
import json
from dataprocess_utils import *
from get_embeddings_function import *
from langchain.vectorstores.chroma import Chroma
from langchain_community.llms.ollama import Ollama
from graph_creation import *
from relations import relations_chemprot
import re


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

all_train_documents_information = train_data_j
all_test_documents_information = test_data_j
all_val_documents_information = val_data_j

# Create readable chunks (assuming the function is defined in dataprocess_utils)
_, test_data, _ = creating_readable_chunks_for_chemprot(all_train_documents_information, all_test_documents_information, all_val_documents_information)

CHROMA_PATH = "chroma_chemprot"
i = 0
model = Ollama(model="llama3relations")
output_for_graph = []


for i in range(1):
    text = test_data[i]["text"]
    correct_label = test_data[i]['label']
    list_of_entities = [f"{ent}" for ent in test_data[i]["entities"]]

    # Get embedding function and search with Chroma
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    search_text = test_data[i]['text'] + f'{list_of_entities}'
    results = db.similarity_search_with_score(test_data[i]['text'], k=5)

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

    # Invoke the model
    import subprocess
    import json
    import base64


    # Directory containing images



    
       
    payload = {
        "model": "llama3relations",
        "prompt": prompt, 
        "stream":False,
    }
    payload = json.dumps(payload)

    curl_command = [
        "curl",
        "-X", "POST",
        "https://f821-106-219-164-23.ngrok-free.app",
        "-H", "Content-Type: application/json",
        "-H", "Authorization: Bearer 2grprGQmjfc77BdX0x5S0wJX8Z8_492i6XzDYLQmCWwEYe14g",
        "-d", payload
    ]

    # Run the curl command and capture the output
    result = subprocess.run(curl_command, capture_output=True, text=True)
    
    output = result.stdout

    print(output)
  


    # response_text = model.invoke(prompt)

    output = json.loads(output)
    print(output["response"])
    predicted_label = output["response"]
    print(correct_label)
    output_for_graph.append([predicted_label, correct_label, list_of_entities])

graph_creation_for_chemprot(output_for_graph)

