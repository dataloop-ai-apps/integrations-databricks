
model_vertical = "events_and_ticketing"
catalog_name = "bitext_innovation_international_fine_tuned_mistral_7b_for_events_and_ticketing"
fine_tuning_data_path = f"/Volumes/{catalog_name}/marketplace/mortgage_loans_data/Mortgage_Loans_train_sample.csv"
version = 2

import mlflow

# Set mlflow registry to databricks-uc
mlflow.set_registry_uri("databricks-uc")

model_name = f"mistral7B-{model_vertical}"

model_mlflow_path = f"models:/{catalog_name}.marketplace.{model_name}/{version}"

model_local_path = f"/tmp/model/{model_name}/"
model_output_local_path = f"/tmp/model/{model_name}_lora_fine_tune"


from datasets import load_dataset

dataset = load_dataset('csv', data_files=fine_tuning_data_path, split="train")

def apply_prompt_template(examples):
  question = examples["question"]
  response = examples["response"]

  return {"text": "<s>[INST] " + question + " [/INST] " + response + "</s>"}

dataset = dataset.map(apply_prompt_template)

print(dataset["text"][556])
