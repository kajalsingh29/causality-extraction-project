import spacy
import json
from dataprocess_utils import *
from get_embeddings_function import *
from langchain.vectorstores.chroma import Chroma
from langchain_community.llms.ollama import Ollama
from graph_creation import *
from relations import relations
import re
CHROMA_PATH = "chroma"

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

# Paths to the JSON data files
train_path = 'C:/Users/USER/Desktop/final_project/BioRED/Train.BioC.JSON'
test_path = 'C:/Users/USER/Desktop/final_project/BioRED/Test.BioC.JSON'
val_path = 'C:/Users/USER/Desktop/final_project/BioRED/Dev.BioC.JSON'

# Load the JSON files
with open(train_path, 'r') as file:
    train_data_j = json.load(file)

with open(test_path, 'r') as file:
    test_data_j = json.load(file)

with open(val_path, 'r') as file:
    val_data_j = json.load(file)

# Extract documents information
all_train_documents_information = train_data_j["documents"]
all_test_documents_information = test_data_j["documents"]
all_val_documents_information = val_data_j["documents"]

# Create readable chunks (assuming the function is defined in dataprocess_utils)
_, test_data, _ = creating_readable_chunks(all_train_documents_information, all_test_documents_information, all_val_documents_information)

# Load the custom spaCy model
nlp_best = spacy.load("model-best")

# Take a test sample
to_text = test_data[1]["text"]

# Process the text with the spaCy model
doc = nlp_best(to_text)

# Iterate over the entities and print the entity text along with its label
list_of_entities = []
for ent in doc.ents:
    print(f"Entity: {ent.text}, Label: {ent.label_}")
    ent_text = f"{ent.text}({ent.label_})"
    # ent_text = ent.text
    list_of_entities.append(ent_text)
print(test_data[1]['text'])


list_of_entities = list(set(list_of_entities))
embedding_function = get_embedding_function()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
results = db.similarity_search_with_score(test_data[1]['text'], k=1)

#PROMPT GENERATION
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

# prompt = """
# You are an expert in extracting relationships from sentences containing biomedical data. Your task is to identify the relationship between all the entities provided to you by taking reference from the sentence.

# ##Instuctions:
# If a single entity has relation with two or more entities mention each relation seperately.

# You have to find the relationship between all the entities mentioned here: 
# {entities}

# Here is the sentence: 
# {sentence}

# These are the relation types you have to choose from:
# {relations}

# Please provide your answer in the following format:
# - Entity 1: [entity1]
# - Entity 2: [entity2]
# - Relation: [relation]

# Note that the sentence might not provide information about a direct relationship between entities. In such cases, you have to infer a relation between them.
# """

# prompt = """
# You are an expert in extracting relationships from sentences containing biomedical data. Your task is to identify the relationship between all the entities provided to you by taking reference from the sentence.

# ##Instuctions:
# If a single entity has relation with two or more entities mention each relation seperately.

# You have to find the relationship between all the entities mentioned here: 
# {entities}

# Here is the sentence: 
# {sentence}

# These are the relation types you have to choose from:
# {relations}
# Positive Correlation between two nodes
# Negative Correlation between two nodes
# Association between two nodes
# Bind between a chemical and a variant
# Drug Interaction between two chemicals
# Cotreatment between two chemicals
# Comparison between two variants
# Conversion between a gene and a gene

# ## Examples:
# Here is the relevant context and examples of how to extract relations from a sentence:
# {examples}

# Please provide your answer in the following format:
# - Entity 1: [entity1]
# - Entity 2: [entity2]
# - Relation: [relation]

# Note that the sentence might not provide information about a direct relationship between entities. In such cases, you have to infer a relation between them.
# """
prompt = """
You are an expert in extracting relationships from sentences containing biomedical data. Your task is to identify the relationship between all the entities provided to you by taking reference from the sentence.

##Instuctions:
If a single entity has relation with two or more entities mention each relation seperately.

You have to find the relationships of **ALL** the entities mentioned here: 
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

Note that the sentence might not provide information about a direct relationship between entities. In such cases, you have to infer a relation between them.

**Make sure to provide relations of all entities mentioned in the sentence**. 
"""

# prompt = """ You are an expert at extraction relations from sentences containing Biomedical data. Relation extraction is to identify the relationship between two entities in a sentence. \n Here is the sentence: {sentence}. You have to find the relationship between these entities present in this sentence. These are the entities present in the sentence between which you have to find relations. ONLY USE THE ENTITIES MENTIONED HERE: {entities}. \nThese are the relation types that you have to choose from {relations}. \nHere is the relevant context and also how you can extract relations from a sentence. {examples}"""


json_template = read_json("output_template.json")
print("USING THIS TEMPLATE", json_template)
prompt = prompt.format(sentence = test_data[1]['text'], entities = list_of_entities,  relations = relations, examples = example, json_template = json_template)

print(prompt)
#LLM INVOKATION
model = Ollama(model="llama3")
response_text = model.invoke(prompt)

file_name = "response_output.txt"
# Write the response_text to the file
with open(file_name, 'w') as file:
    file.write(response_text)
print(f"Response text has been written to {file_name}.")

output_json = load_json(response_text)
graph_creation(output_json)


