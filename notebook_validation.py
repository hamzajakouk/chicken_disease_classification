import nbformat

def validate_notebook(notebook_path):
    try:
        nbformat.read(notebook_path, as_version=nbformat.NO_CONVERT)
        print("Notebook is valid.")
    except nbformat.ValidationError as e:
        print("Notebook is invalid. Validation error:")
        print(e)

# Call the function with the path to your notebook file
validate_notebook("/home/azureuser/chicken_disease_classification/research/01_data_ingestion.ipynb")
