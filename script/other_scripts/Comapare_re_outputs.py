import json

def extract_sro_triples(json_data):
    """
    Extract Subject-Relation-Object (SRO) triples from the JSON data.
    """
    sro_triples = []
    for relation in json_data.get("relations", []):
        subject = relation.get("subject_entity", {}).get("name", "Unknown Subject")
        predicate = relation.get("predicate", "Unknown Predicate")
        obj = relation.get("object_entity", {}).get("name", "Unknown Object")
        sro_triples.append((subject, predicate, obj))
    # Sort triples by subject
    sro_triples.sort(key=lambda x: x[0])
    return sro_triples

def compare_sro_triples(triples1, triples2):
    """
    Compare two lists of SRO triples and return the differences.
    """
    set1 = set(triples1)
    set2 = set(triples2)
    
    only_in_file1 = set1 - set2
    only_in_file2 = set2 - set1
    common_triples = set1 & set2
    
    return only_in_file1, only_in_file2, common_triples

def main():
    # Input JSON file paths
    file1 = "/home/dolor/Documents/Aim2/output/re/processed/PMC7384185_gpt4.json"
    file2 = "/home/dolor/Documents/Aim2/output/re/processed/PMC7384185_local.json"
    
    # Load JSON data from files
    try:
        with open(file1, 'r') as f1, open(file2, 'r') as f2:
            data1 = json.load(f1)
            data2 = json.load(f2)
    except Exception as e:
        print(f"Error reading files: {e}")
        return
    
    # Extract SRO triples
    triples1 = extract_sro_triples(data1)
    triples2 = extract_sro_triples(data2)
    
    # Compare SRO triples
    only_in_file1, only_in_file2, common_triples = compare_sro_triples(triples1, triples2)
    
    # Print results
    counter1 = len(triples1)
    counter2 = len(triples2)
    counter3 = len(common_triples)
    print(f"Number of relations in first file: {counter1}")
    print(f"Number of relations in second file: {counter2}")
    print(f"Number of common relations: {counter3}")
    
    print("\nSRO Triples only in the first file:")
    for triple in sorted(only_in_file1, key=lambda x: x[0]):
        print(triple)
    
    print("\nSRO Triples only in the second file:")
    for triple in sorted(only_in_file2, key=lambda x: x[0]):
        print(triple)
    
    print("\nCommon SRO Triples in both files:")
    for triple in sorted(common_triples, key=lambda x: x[0]):
        print(triple)

if __name__ == "__main__":
    main()