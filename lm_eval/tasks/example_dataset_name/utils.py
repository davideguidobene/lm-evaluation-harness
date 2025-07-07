import datasets

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    print("Processing dataset...")
    print(type(dataset))
    print(dataset)
    def _process_doc(doc):
        #ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        out_doc = {
            "query": "Say 1", #f'Say whether the following text is positive, negative or neutral (and do not write any other word): \"{doc["text"]}\"', #+ ": " + ctx,
            #"choices": [preprocess(ending) for ending in doc["endings"]],
            "gold": 1 #doc["label"],
        }
        return out_doc

    return dataset.map(_process_doc)


def my_custom_accuracy(preds, refs):
    print("Calculating custom accuracy...")
    print(preds)
    print("Printing refs:")
    print(refs)
    return {"my_custom_accuracy": 1.0}

def process_results(doc, results):
    print("Processing results...")
    print(results)
    print("Now printing the doc:")
    print(doc)
    return {
        "acc": 1,
        "acc_norm": 1
    }
