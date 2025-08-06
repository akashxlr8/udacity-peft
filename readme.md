# PEFT Project Notebook Description

## Project Overview
This notebook demonstrates parameter-efficient fine-tuning (PEFT) of a pre-trained language model for a sequence classification task using the LoRA (Low-Rank Adaptation) technique. The workflow follows the assignment requirements:

- Load and evaluate a pre-trained foundation model
- Perform parameter-efficient fine-tuning using LoRA
- Evaluate and compare the performance of the original and fine-tuned models

## Choices and Rationale
- **PEFT Technique:** LoRA was selected for its compatibility and efficiency with transformer models.
- **Model:** GPT-2 was chosen as the base model due to its small size and support for sequence classification tasks.
- **Dataset:** The Rotten Tomatoes dataset from Hugging Face Datasets library was used, as it is suitable for text classification and small enough for efficient experimentation.
- **Evaluation:** Model performance is measured using accuracy, computed via the Hugging Face `evaluate` library and the Trainer API.

## Notebook Steps
1. **Setup and Installation**
   - Install required libraries: `transformers`, `torch`, `peft`, `datasets`, `evaluate`.
2. **Load Tokenizer and Dataset**
   - Load the GPT-2 tokenizer and Rotten Tomatoes dataset.
   - Preprocess and tokenize the dataset for sequence classification.
3. **Evaluate the Foundation Model**
   - Load the pre-trained GPT-2 model for sequence classification.
   - Evaluate its accuracy on the test set before fine-tuning.
4. **Parameter-Efficient Fine-Tuning (LoRA)**
   - Create a LoRA configuration and apply it to the base model.
   - Fine-tune the model on the training set using the Hugging Face Trainer.
   - Save the fine-tuned LoRA adapter.
5. **Evaluate the Fine-Tuned Model**
   - Load the fine-tuned LoRA model.
   - Evaluate its accuracy on the test set.
6. **Performance Comparison**
   - Compare and report the accuracy of the original and fine-tuned models.

## Submission Checklist
-  Jupyter notebook file with all code and results
-  Saved model weights (LoRA adapter)

## Notes
- The notebook is modular and well-logged for clarity and reproducibility.
- All steps are compatible with the Udacity Workspace and can be run on CPU or GPU.
- The code can be easily adapted for other models, datasets, or PEFT techniques.
