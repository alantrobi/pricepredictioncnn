import os
import shutil

KEEP_FOLDER = "venv"

def clean_project():

    current_dir = os.getcwd()
    print(f"Cleaning project: {current_dir}\n")

    for item in os.listdir(current_dir):

        item_path = os.path.join(current_dir, item)

        # Skip venv
        if item == KEEP_FOLDER:

            continue

        # ----------------------------
        # Delete folders
        # ----------------------------
        if os.path.isdir(item_path):
            try:
                shutil.rmtree(item_path)
                print(f"Deleted folder: {item}")
            except Exception as e:
                print(f"Error deleting folder {item}: {e}")

        # ----------------------------
        # Delete .keras files
        # ----------------------------
        elif os.path.isfile(item_path) and item.endswith(".keras"):
            try:
                os.remove(item_path)
                print(f"Deleted model file: {item}")
            except Exception as e:
                print(f"Error deleting file {item}: {e}")


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":

    confirm = input("Delete all folders (except venv) and all .keras files? (y/n): ")

    if confirm.lower() == 'y':
        clean_project()
        print("\n✅ Cleanup complete.")
    else:
        print("Aborted.")