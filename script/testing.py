import json
def main():
    # parse the json for metabolites and store them into a dictionary
    file_input = "output/PMC10374582.json"

    # read the file and only take compounds
    with open(file_input, "r") as f:
        data = json.load(f)
        compounds = set()
        for i in range(len(data)):
            item = data[i]
            for compound in item["compounds"]:
                if compound["name"] not in compounds:
                    compounds.add(compound["name"])

    # Do something with the compounds dictionary sorted
    sorted_compounds = sorted(compounds)
    for compound in sorted_compounds:
        print(compound)


if __name__ == "__main__":
    main()
