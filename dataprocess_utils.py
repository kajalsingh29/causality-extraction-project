import json
def find_entity_name(entity_id, data):

    # Check relations to find if the entity_id is involved
    for passage in data['passages']:
      for annotation in passage['annotations']:
        if annotation['infons']['identifier'] == entity_id:
          return annotation['text']

    return None  # If not found



def creating_sample(document) -> list:
  """
  The function takes in input the element with key passages
  """
  training_sample = []
  for elem in document['passages']:
    temp_dict = {}
    temp_dict['text'] = elem['text']
    temp_dict['entities'] = []
    starting_point = elem['offset']
    for annotation in elem['annotations']:
      start = annotation['locations'][0]['offset'] - starting_point
      end = start + annotation['locations'][0]['length']
      text_label = annotation['text']
      assert elem['text'][start:end] == text_label # checking the correct extraction
      label = annotation['infons']['type'].upper()
      temp_dict['entities'].append((elem['text'][start:end], label))
    temp_dict['relations'] = []
    for rels in document['relations']:
      current_relation = rels['infons']
      entity1_id= current_relation['entity1']
      entity2_id = current_relation['entity2']
      relation_type = current_relation['type']
      entity_name1 = find_entity_name(entity1_id, document)
      entity_name2 = find_entity_name(entity2_id, document)
      temp_dict['relations'].append((entity_name1, entity_name2, relation_type))
    training_sample.append(temp_dict)
  
  
  
  return training_sample

def creating_sample_for_chemprot(element):
  import re
  element = json.loads(element)
  extracted_double_angle_brackets = re.findall(r'<<\s*(.*?)\s*>>', element['text'])
  extracted_double_square_brackets = re.findall(r'\[\[\s*(.*?)\s*]]', element['text'])
  
  element["entities"] = [extracted_double_angle_brackets[0], extracted_double_square_brackets[0]]
  return element


def creating_readable_chunks(all_train_documents_information, all_test_documents_information, all_val_documents_information):
  training_data = []
  
  for document in all_train_documents_information:

    training_sample = creating_sample(document)
    training_data += training_sample

  test_data = []
  for document in all_test_documents_information:
    test_sample = creating_sample(document)
    test_data += test_sample

  val_data = []
  for document in all_val_documents_information:
    val_sample = creating_sample(document)
    val_data += val_sample


  return training_data, test_data, val_data


# train_path = 'C:/Users/USER/Desktop/final_project/BioRED/Train.BioC.JSON'
# test_path = 'C:/Users/USER/Desktop/final_project/BioRED/Test.BioC.JSON'
# val_path = 'C:/Users/USER/Desktop/final_project/BioRED/Dev.BioC.JSON'

# with open(train_path, 'r') as file:
#     train_data_j = json.load(file)

# with open(test_path, 'r') as file:
#     test_data_j = json.load(file)

# with open(val_path, 'r') as file:
#     val_data_j = json.load(file)

# all_train_documents_information = train_data_j["documents"]
# all_test_documents_information = test_data_j["documents"]
# all_val_documents_information = val_data_j["documents"]

# creating_readable_chunks(all_train_documents_information,all_test_documents_information,all_val_documents_information)


def creating_readable_chunks_for_chemprot(all_train_documents_information, all_test_documents_information, all_val_documents_information):
  training_data = []
  
  for element in all_train_documents_information:

    training_sample = creating_sample_for_chemprot(element)
    
    training_data.append(training_sample)
 
  test_data = []
  for element in all_test_documents_information:
    test_sample = creating_sample_for_chemprot(element)
    test_data.append(test_sample)

  val_data = []
  for element in all_val_documents_information:
    val_sample = creating_sample_for_chemprot(element)
    val_data.append(val_sample)

  return training_data, test_data, val_data
