import datasets

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    print("Processing dataset...")
    print(type(dataset))
    print(dataset)
    def _process_doc(doc):
        print("Printing doc inside procces_docs:")
        print(doc)
        #ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        out_doc = {
            "query": f'Say whether the following text is positive, negative or neutral: \"{doc["text"]}\"', #+ ": " + ctx,
            #"choices": [preprocess(ending) for ending in doc["endings"]],
            "gold": doc["label"],
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

import re
from lm_eval.api.instance import Instance
from lm_eval.api.task import ConfigurableTask

# A simple function to determine the system message based on the doc
def get_system_message_for_doc(doc):
    # Example: if the document is about math, use a math-specific system prompt.
    # Otherwise, use a general one.
    if "math" in doc.get("subject", "").lower():
        return "You are a world-class mathematician. Solve the following problem with detailed steps."
    else:
        return "You are a helpful and accurate assistant. Please answer the question."

'''
class MyCustomTask(ConfigurableTask):
    # ... (other required methods like has_training_docs, doc_to_text, etc.)

    def construct_requests(self, doc, ctx, **kwargs):
        """
        This method constructs the final requests that are sent to the model.
        We will override it to prepend our dynamic system message.
        """
        # 1. Get the dynamic system message for the current document
        system_message = get_system_message_for_doc(doc)
        
        # 2. Prepend the system message to the context (ctx).
        # The 'ctx' variable already contains the formatted few-shot examples and the current query.
        # Note: We add a newline for clean separation.
        final_prompt = f"{system_message}\n\n{ctx}"

        # 3. Create the request using the modified prompt.
        # We'll use generate_until for an open-ended generation task.
        return [
            Instance(
                request_type="generate_until",
                doc=doc,
                arguments=(final_prompt, {"until": ["\n\n"], "max_gen_toks": 256}),
                idx=0,
                **kwargs,
            )
        ]

    def process_results(self, doc, results):
        # ... your evaluation logic
        return { "accuracy": 0.0 } # Placeholder

from lm_eval.models.huggingface import HFLM
class CustomHFLM(HFLM):
    pass


from lm_eval.models import MODEL_REGISTRY
MODEL_REGISTRY["hf"] = CustomHFLM
'''
