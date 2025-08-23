import os


def cleanup_SAP2000():
    # Get StructGNN folder (parent of MP_TESTING)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    structgnn_folder = base_dir  # StructGNN path

    for filename in os.listdir(structgnn_folder):
        filepath = os.path.join(structgnn_folder, filename)

        #Delete all all files with 3DFrame in the name
        if os.path.isfile(filepath) and "3DFrame" in filename:
            os.remove(filepath)
            print(f"Deleted: {filepath}")

def cleanup_csv():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    structgnn_folder = base_dir  # StructGNN path

    for filename in os.listdir(structgnn_folder):
        filepath = os.path.join(structgnn_folder, filename)

        # Delete all files with .csv at the end
        if os.path.isfile(filepath) and filename.endswith(".csv"):
            os.remove(filepath)
            print(f"Deleted: {filepath}")


if __name__ == "__main__":
    cleanup_SAP2000()
    cleanup_csv()