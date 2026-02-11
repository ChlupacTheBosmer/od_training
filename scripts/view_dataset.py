import argparse
import fiftyone as fo
import os

def check_dataset_exists(name):
    return name in fo.list_datasets()

def main():
    parser = argparse.ArgumentParser(description="View FiftyOne Dataset")
    parser.add_argument("name", help="Name of the FiftyOne dataset to view")
    parser.add_argument("--import-dir", help="Optional: Import YOLO dataset from this directory if it doesn't exist.")
    
    args = parser.parse_args()

    if check_dataset_exists(args.name):
        print(f"Loading existing dataset: {args.name}")
        dataset = fo.load_dataset(args.name)
    elif args.import_dir:
        print(f"Dataset {args.name} not found. Importing from {args.import_dir}...")
        
        dataset_dir = os.path.abspath(args.import_dir)
        yaml_path = None
        if os.path.isfile(dataset_dir):
            yaml_path = dataset_dir
            dataset_dir = os.path.dirname(dataset_dir)
            
        kwargs = {}
        if yaml_path:
            kwargs["yaml_path"] = yaml_path
            
        dataset = fo.Dataset.from_dir(
            dataset_dir=dataset_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            name=args.name,
            **kwargs
        )
    else:
        print(f"Error: Dataset '{args.name}' not found and no --import-dir specified.")
        print("Available datasets:", fo.list_datasets())
        return

    print("Launching App...")
    session = fo.launch_app(dataset)
    session.wait()

if __name__ == "__main__":
    main()
