def save_json_file(output_path, output_name, output):
    import json
    import os
    
    file_path = output_path + '/' + output_name + '.json' 
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    with open(file_path, 'w') as file:
        file.write(json.dumps(output, indent = 4, sort_keys = True))
    
    return

def save_model(output_path, model_name, model):
    import json
    import os
        
    json_string = model.to_json()  ### getting json of model architecture as in string
    
    model_json = json.loads(json_string)  ### converting json string to json (dict)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    with open(output_path + '/' + model_name + '_architecture.json', 'w') as file:
        json.dump(model_json, file, indent = 4, sort_keys = True)
        
    model.save(output_path + '/' + model_name + '.h5')
      
    return    