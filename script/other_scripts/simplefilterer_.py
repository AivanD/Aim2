import json

# filepath: /home/aivan/Aim2/output/re/PMC8164654_local_simple.json
input_file = "/home/aivan/Aim2/output/re/PMC8164654_local_simple.json"
output_file = "/home/aivan/Aim2/output/re/filtered_PMC8164654_local_simple.json"

def filter_json(input_path, output_path):
    # Load the JSON data
    with open(input_path, "r") as file:
        data = json.load(file)
    
    # Filter the data
    filtered_data = [
        entry for entry in data
        if "subject_cid" in entry and entry["subject_cid"] is not None
        and "object_ontology_id" in entry and entry["object_ontology_id"] is not None
    ]
    
    # Save the filtered data to a new file
    with open(output_path, "w") as file:
        json.dump(filtered_data, file, indent=2)
    
    print(f"Filtered data saved to {output_path}")

# Run the function
filter_json(input_file, output_file)