import spacy
import json
from dataprocess_utils import *
from get_embeddings_function import *
from langchain.vectorstores.chroma import Chroma
from langchain_community.llms.ollama import Ollama
from graph_creation import *
from relations import relations_chemprot
import re
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from tqdm import tqdm
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
file_name = "response_output.txt"

model = Ollama(model="llama3")

true_labels = []
predicted_labels = []
print(len(test_data[:1000]))
for i in tqdm(range(len(test_data[:1000])), desc="Processing test data"):
    text = test_data[i]["text"]
    correct_label = test_data[i]['label']
    list_of_entities=[]

    for ent in test_data[i]["entities"]:
        
        ent_text = f"{ent}"
        # ent_text = ent.text
        list_of_entities.append(ent_text)


    embedding_function = get_embedding_function()

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_score(test_data[i]['text'], k=5)

    EXAMPLES = """\nExample: {Number} \nSentence: {sentence} \nEntities: {entities} \nRelations: {relations}"""
    example = ""
    cnt = 1
    for document, score in results:
        EXAMPLES = EXAMPLES.format(Number = cnt, sentence = document.page_content, entities =  document.metadata['entities'], relations = document.metadata['relations'])
        example = example + EXAMPLES
        EXAMPLES = """
        Example: {Number}
        Text: {sentence}
        Entities: {entities}
        Relations: {relations}
        """
        cnt += 1



    prompt = """
    You are an expert in extracting relationships from sentences containing biomedical data. Your task is to identify the relation between the entities provided to you by taking reference from the sentence.

    ##Instuctions:

    You have to find the relation between the entities mentioned here: 
    {entities}

    Here is the sentence: 
    {sentence}

    These are the relation types you have to choose from:
    {relations}

    ## Examples:
    Here is the relevant context and examples of how to extract relations from a sentence:
    {examples}

    Please provide your answer in the following JSON format. Include explanation of why entity1 is related to entity2 for your better understanding:
    {json_template}
    
    Note that the sentence might not provide information about a direct relationship between entities. In such cases, you have to infer a relation between them by using information from examples.

    """

    # prompt = """ You are an expert at extraction relations from sentences containing Biomedical data. Relation extraction is to identify the relationship between two entities in a sentence. \n Here is the sentence: {sentence}. You have to find the relationship between these entities present in this sentence. These are the entities present in the sentence between which you have to find relations. ONLY USE THE ENTITIES MENTIONED HERE: {entities}. \nThese are the relation types that you have to choose from {relations}. \nHere is the relevant context and also how you can extract relations from a sentence. {examples}"""


    json_template = read_json("output_template.json")
 
    prompt = prompt.format(sentence = test_data[i]['text'], entities = list_of_entities,  relations = relations_chemprot, examples = example, json_template = json_template)

    
    response_text = model.invoke(prompt)
    with open(file_name, 'a') as file:
        file.write(f"Response for test data {i}:\n")
        file.write(response_text)
        file.write("\n\n")  # Add spacing between entries
    try:
        # Try to load the model's output as JSON
        output_json = load_json(response_text)

        # Get the predicted label
        predicted_label = output_json["Relation"]
     
        # Append true and predicted labels to the list
        true_labels.append(correct_label)
        predicted_labels.append(predicted_label)
    
    except json.JSONDecodeError:
        print(f"Error decoding JSON for index {i}, skipping this instance.")
        continue  # Skip to the next iteration if there's a JSON error
    
    except KeyError:
        print(f"Missing 'Relation' field in JSON response for index {i}, skipping this instance.")
        continue  # Skip if 'Relation' field is missing

# Calculate the F1 score
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Accuracy: {accuracy}")

# Precision (with 'micro' averaging)
precision = precision_score(true_labels, predicted_labels, average='micro')
print(f"Precision: {precision}")

# Recall (with 'micro' averaging)
recall = recall_score(true_labels, predicted_labels, average='micro')
print(f"Recall: {recall}")

f1 = f1_score(true_labels, predicted_labels, average='micro')  # 'micro' computes global metrics
print(f"F1 Score: {f1}")