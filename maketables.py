from py_markdown_table.markdown_table import markdown_table

def mk(title, rows):
    print(f"\n## {title}\n")
    print("```text")
    print(markdown_table(rows).get_markdown())
    print("```\n")

# System Requirements
mk("System Requirements (PyPI)", [
    {"Component": "OS", "Tested Version": "Windows 10"},
    {"Component": "Python", "Tested Version": "3.9.19"},
    {"Component": "CUDA", "Tested Version": "12.6"},
])

# Parameters
mk("Parameters (PyPI)", [
    {"Parameter": "`train`", "Type": "pandas.DataFrame", "Description": "Training dataframe containing text, label, and tabular feature columns"},
    {"Parameter": "`val`", "Type": "pandas.DataFrame", "Description": "Validation dataframe with same structure as train"},
    {"Parameter": "`pretrained_model_name`", "Type": "str", "Description": "Base model to fine-tune. Supports: `emilyalsentzer/Bio_ClinicalBERT` or `microsoft/biogpt`"},
    {"Parameter": "`label_col`", "Type": "str", "Description": "Column name of the binary outcome label (must contain True/False values)"},
    {"Parameter": "`text_col`", "Type": "str", "Description": "Column name containing the clinical text"},
    {"Parameter": "`columns_unique_labels_of_tabular_features`", "Type": "dict", "Description": "Map feature → num unique values (use 1 continuous, >1 categorical)"},
    {"Parameter": "`lr`", "Type": "float", "Description": "Learning rate (default: 0.001)"},
    {"Parameter": "`num_epochs`", "Type": "int", "Description": "Epochs (default: 5)"},
    {"Parameter": "`lr_of_tabular_infused_features`", "Type": "float", "Description": "LR for tabular pre-training (default: 0.0001)"},
])

# Returns
mk("Returns (PyPI)", [
    {"Return": "`model`", "Type": "PeftModel", "Description": "The trained IA3 model"},
    {"Return": "`tokenizer`", "Type": "AutoTokenizer", "Description": "The tokenizer for the model"},
])
