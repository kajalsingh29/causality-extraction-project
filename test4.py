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
model = Ollama(model="llama3")
output_for_graph = []

text = "Kinetics of inhibition of << human and rat dihydroorotate dehydrogenase >> by atovaquone, [[ lawsone ]] derivatives, brequinar sodium and polyporic acid."

list_of_entities = ['human and rat dihydroorotate dehydrogenase', 'lawsone']

# Get embedding function and search with Chroma
embedding_function = get_embedding_function()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
search_text = text + f'{list_of_entities}'
results = db.similarity_search_with_score(text, k=5)

EXAMPLES = """\nExample: {Number} \Text: {sentence} \nEntities: {entities} \nRelations: {relations}"""
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
list_of_relations = {
'PRODUCT-OF': 'The molecule is the result of a biochemical reaction catalyzed by an enzyme or process.',
'AGONIST-ACTIVATOR': 'A molecule that binds to a receptor and enhances its activity or signaling response.',
'INHIBITOR': 'A substance that reduces or blocks the activity of an enzyme or receptor.',
'ANTAGONIST': 'A molecule that binds to a receptor but prevents it from producing its biological response.',
'ACTIVATOR': 'A molecule that increases the activity of an enzyme or a receptor.',
'AGONIST-INHIBITOR': 'A molecule that can both stimulate and inhibit the activity of a receptor or enzyme, depending on context.',
'DOWNREGULATOR': 'A molecule that decreases the expression or activity of a protein or receptor.',
'AGONIST': 'A molecule that binds to and activates a receptor to produce a biological response.',
'SUBSTRATE_PRODUCT-OF': 'A compound that acts as a substrate in one reaction and is also the product of another reaction.',
'UPREGULATOR': 'A molecule that increases the expression or activity of a protein or receptor.',
'INDIRECT-UPREGULATOR': 'A molecule that increases the activity or expression of a target through an indirect mechanism.',
'INDIRECT-DOWNREGULATOR': 'A molecule that decreases the activity or expression of a target through an indirect mechanism.',
'SUBSTRATE': 'A molecule upon which an enzyme acts to produce a product.'
}
# Create the prompt
prompt = f"""
    You are an expert in extracting relationships from sentences containing biomedical data. Your task is to identify the relation between the entities provided to you by taking reference from the sentence.

##Instructions:

You have to find the relation between the entities mentioned here: 
{list_of_entities}

Here is the sentence: 
{text}

These are the relation types you have to choose from. I have added explaination for each relation to help you in identifying relationships:
{list_of_relations}

## Context:
Here is the relevant context to help you understand the relation between the entities:
{example}

You have to infer the relation between the entities by making sure that the relation's description matches with the relation between the actual entities

Please provide your answer in the following JSON format. Use chain of thought to infer the relations between entities:
{read_json("output_template.json")}


"""
print(prompt)
# # Invoke the model
# response_text = model.invoke(prompt)
# print(response_text)
# output_json = load_json(response_text)

# print(correct_label)
