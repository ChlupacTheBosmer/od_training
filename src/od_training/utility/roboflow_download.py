#!/usr/bin/env python3
"""Download datasets from Roboflow with optional FiftyOne import.

This script downloads datasets from the Roboflow platform and optionally imports
them into FiftyOne with automatic split-based tagging.

Configuration Priority (highest to lowest):
- Command-line arguments
- local_config.json
- Environment variables

Features:
- Download datasets in any Roboflow-supported format
- Automatic unzipping with optional cleanup
- Import to FiftyOne with split-based tagging (train/val/test)
- Graceful error handling for unsupported formats
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path
from typing import Optional


from .runtime_config import (
    ensure_local_config,
    get_roboflow_api_key,
    get_roboflow_default,
    get_data_dir,
)

# Import FiftyOne only if needed (lazy import)
try:
    import fiftyone as fo
    FIFTYONE_AVAILABLE = True
except ImportError:
    FIFTYONE_AVAILABLE = False

# Import roboflow only if needed (lazy import)
try:
    from roboflow import Roboflow
    ROBOFLOW_AVAILABLE = True
except ImportError:
    ROBOFLOW_AVAILABLE = False


def prompt_with_default(prompt: str, default: Optional[str] = None, required: bool = True) -> str:
    """Prompt user for input with optional default value.
    
    Args:
        prompt: The prompt message
        default: Default value (shown to user if provided)
        required: If True, will keep prompting until a value is provided
        
    Returns:
        User input or default value
    """
    if default:
        full_prompt = f"{prompt} [{default}]: "
    else:
        full_prompt = f"{prompt}: "
    
    while True:
        user_input = input(full_prompt).strip()
        
        if user_input:
            return user_input
        elif default:
            return default
        elif not required:
            return ""
        else:
            print("This field is required. Please provide a value.")


def get_interactive_params(args, config_path: Path | None = None):
    """Get parameters interactively with config/env fallbacks.
    
    Shows user what values are available from config/env so they can choose
    to use them or override.
    """
    print("\n=== Interactive Configuration ===")
    print("Press Enter to use the shown default value (if any).\n")
    
    # API Key
    api_key_default = None
    try:
        api_key_default = get_roboflow_api_key(args.api_key, config_path=config_path)
        print(f"✓ API key found in config/env")
    except ValueError:
        print("✗ No API key found in config/env")
    
    if not api_key_default:
        args.api_key = prompt_with_default("Roboflow API Key", required=True)
    else:
        args.api_key = api_key_default
    
    # Workspace
    workspace_default = (
        get_roboflow_default("workspace", config_path=config_path)
        if not args.workspace
        else args.workspace
    )
    if workspace_default:
        print(f"✓ Workspace found in config/env: {workspace_default}")
    else:
        print("✗ No workspace found in config/env")
    args.workspace = prompt_with_default("Workspace ID", workspace_default, required=True)
    
    # Project
    project_default = (
        get_roboflow_default("project", config_path=config_path)
        if not args.project
        else args.project
    )
    if project_default:
        print(f"✓ Project found in config/env: {project_default}")
    else:
        print("✗ No project found in config/env")
    args.project = prompt_with_default("Project ID", project_default, required=True)
    
    # Version (always required, no config default)
    if not args.version:
        args.version = int(prompt_with_default("Dataset Version (number)", required=True))
    
    # Format (always required, no config default)
    if not args.format:
        args.format = prompt_with_default("Dataset Format", default="yolov11", required=True)
    
    # Download directory
    data_dir = get_data_dir(config_path=config_path)
    default_download_dir = os.path.join(data_dir, "roboflow", args.project)
    if not args.download_dir:
        print(f"✓ Default download directory from config: {default_download_dir}")
        args.download_dir = prompt_with_default("Download Directory", default_download_dir, required=False)
        if not args.download_dir:
            args.download_dir = default_download_dir
    
    print()
    return args


def download_dataset(api_key: str, workspace: str, project: str, version: int, 
                     format_type: str, download_dir: str) -> str:
    """Download dataset from Roboflow.
    
    Args:
        api_key: Roboflow API key
        workspace: Workspace ID
        project: Project ID
        version: Dataset version number
        format_type: Download format (e.g., 'yolov11', 'coco')
        download_dir: Directory to download into
        
    Returns:
        Path to downloaded dataset directory
    """
    if not ROBOFLOW_AVAILABLE:
        raise ImportError(
            "roboflow package is not installed. Install it with: pip install roboflow"
        )
    
    print(f"\n🔄 Connecting to Roboflow...")
    rf = Roboflow(api_key=api_key)
    
    print(f"📦 Accessing project: {workspace}/{project}")
    proj = rf.workspace(workspace).project(project)
    
    print(f"📋 Fetching version {version}...")
    dataset_version = proj.version(version)
    
    # Set location for download
    os.makedirs(download_dir, exist_ok=True)
    original_dir = os.getcwd()
    
    try:
        os.chdir(download_dir)
        print(f"⬇️  Downloading in format '{format_type}' to: {download_dir}")
        
        dataset = dataset_version.download(format_type)
        
        # The download() method returns an object with .location attribute
        if hasattr(dataset, 'location'):
            dataset_path = dataset.location
        else:
            # Fallback: assume it downloaded to current directory
            dataset_path = os.path.join(download_dir, project)
        
        print(f"✅ Download complete: {dataset_path}")
        return dataset_path
        
    finally:
        os.chdir(original_dir)


def unzip_dataset(download_dir: str) -> Optional[str]:
    """Find and unzip dataset archive in download directory.
    
    Args:
        download_dir: Directory containing the downloaded archive
        
    Returns:
        Path to extracted directory or None if no zip found
    """
    zip_files = list(Path(download_dir).glob("*.zip"))
    
    if not zip_files:
        print("ℹ️  No zip file found (dataset may already be extracted)")
        return None
    
    zip_file = zip_files[0]
    print(f"📂 Unzipping: {zip_file.name}")
    
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(download_dir)
    
    print(f"✅ Extracted to: {download_dir}")
    return str(zip_file)


def import_to_fiftyone(dataset_path: str, format_type: str, dataset_name: str,
                       train_tag: str, val_tag: str, test_tag: str) -> bool:
    """Import downloaded dataset into FiftyOne with split-based tagging.
    
    Args:
        dataset_path: Path to the downloaded dataset
        format_type: Format of the dataset (e.g., 'yolov11', 'coco')
        dataset_name: Name for the FiftyOne dataset
        train_tag: Tag for training samples
        val_tag: Tag for validation samples
        test_tag: Tag for test samples
        
    Returns:
        True if successful, False otherwise
    """
    if not FIFTYONE_AVAILABLE:
        print("❌ FiftyOne is not installed. Cannot import dataset.")
        print("   Install with: pip install fiftyone")
        return False
    
    # Determine if format is supported
    supported_formats = {
        "yolov5": fo.types.YOLOv5Dataset,
        "yolov8": fo.types.YOLOv5Dataset,  # Same format as v5
        "yolov11": fo.types.YOLOv5Dataset,  # Same format as v5
        "coco": fo.types.COCODetectionDataset,
    }
    
    format_lower = format_type.lower()
    dataset_type = supported_formats.get(format_lower)
    
    if not dataset_type:
        print(f"⚠️  Format '{format_type}' is not directly supported for FiftyOne import.")
        print(f"   Supported formats: {', '.join(supported_formats.keys())}")
        print(f"   You'll need to import this dataset manually.")
        return False
    
    try:
        print(f"\n📥 Importing to FiftyOne as '{dataset_name}'...")
        
        # Check if dataset already exists
        if dataset_name in fo.list_datasets():
            print(f"⚠️  Dataset '{dataset_name}' already exists in FiftyOne.")
            overwrite = input("   Overwrite? (y/n): ").strip().lower()
            if overwrite == 'y':
                fo.delete_dataset(dataset_name)
            else:
                print("   Import cancelled.")
                return False
        
        # For YOLO formats, look for data.yaml or dataset.yaml
        if format_lower in ["yolov5", "yolov8", "yolov11"]:
            yaml_path = None
            for yaml_name in ["dataset.yaml", "data.yaml"]:
                potential_path = os.path.join(dataset_path, yaml_name)
                if os.path.exists(potential_path):
                    yaml_path = potential_path
                    break
            
            if yaml_path:
                dataset = fo.Dataset.from_dir(
                    dataset_dir=dataset_path,
                    dataset_type=dataset_type,
                    name=dataset_name,
                    yaml_path=yaml_path,
                )
            else:
                dataset = fo.Dataset.from_dir(
                    dataset_dir=dataset_path,
                    dataset_type=dataset_type,
                    name=dataset_name,
                )
            
            # Tag samples based on their location
            # YOLO datasets typically have train/, valid/, test/ directories
            split_map = {
                "train": train_tag,
                "valid": val_tag,
                "val": val_tag,
                "test": test_tag,
            }
            
            for sample in dataset:
                # Determine split from filepath
                filepath = sample.filepath
                for split_dir, tag in split_map.items():
                    if f"/{split_dir}/" in filepath or f"\\{split_dir}\\" in filepath:
                        sample.tags.append(tag)
                        break
            
            dataset.save()
            
        elif format_lower == "coco":
            # COCO format typically has train, val, test JSON files
            # This is more complex and depends on Roboflow's COCO structure
            # For simplicity, import the whole dataset and let user handle splits
            print("⚠️  COCO format detected. Importing entire dataset without automatic split tagging.")
            print("   You may need to tag samples manually based on the annotation files.")
            
            dataset = fo.Dataset.from_dir(
                dataset_dir=dataset_path,
                dataset_type=dataset_type,
                name=dataset_name,
            )
        
        print(f"✅ Successfully imported {len(dataset)} samples to FiftyOne")
        print(f"   Dataset name: {dataset_name}")
        
        # Show tag distribution
        tag_counts = dataset.count_sample_tags()
        if tag_counts:
            print(f"   Tag distribution: {dict(tag_counts)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to import to FiftyOne: {e}")
        import traceback
        traceback.print_exc()
        return False


def main(argv=None):
    """Download a Roboflow dataset and optionally import it into FiftyOne.

    Args:
        argv: Optional argument list. Uses ``sys.argv`` when omitted.

    Returns:
        Exit code ``0`` on success. Exits with non-zero status on failures.
    """
    parser = argparse.ArgumentParser(
        description="Download datasets from Roboflow with optional FiftyOne import.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (will prompt for all parameters)
  python download_roboflow.py --interactive

  # Download with CLI arguments
  python download_roboflow.py \\
    --api-key YOUR_API_KEY \\
    --workspace pests-detection-q0oyd \\
    --project mice_all \\
    --version 2 \\
    --format yolov11

  # Download and import to FiftyOne
  python download_roboflow.py \\
    --workspace pests-detection-q0oyd \\
    --project mice_all \\
    --version 2 \\
    --format yolov11 \\
    --import-fiftyone \\
    --fiftyone-name mice_dataset

Configuration:
  API key, workspace, and project can be configured in:
    config/local_config.json
  
  Priority: CLI args > config file > environment variables
"""
    )
    
    # API & Project parameters
    parser.add_argument("--api-key", help="Roboflow API key (or set in config/env)")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to local config JSON (otherwise uses ODT_CONFIG_PATH/default lookup).",
    )
    parser.add_argument("--workspace", help="Roboflow workspace ID (or set in config/env)")
    parser.add_argument("--project", help="Roboflow project ID (or set in config/env)")
    parser.add_argument("--version", type=int, help="Dataset version number (required)")
    parser.add_argument("--format", help="Download format (e.g., 'yolov11', 'coco')")
    
    # Download options
    parser.add_argument("--download-dir", help="Directory to download dataset (default: data/roboflow/<project>)")
    parser.add_argument("--unzip", action="store_true", default=True, help="Unzip downloaded archive (default: True)")
    parser.add_argument("--no-unzip", dest="unzip", action="store_false", help="Don't unzip the archive")
    parser.add_argument("--delete-zip", action="store_true", help="Delete zip file after extraction")
    
    # FiftyOne options
    parser.add_argument("--import-fiftyone", action="store_true", help="Import dataset into FiftyOne")
    parser.add_argument("--fiftyone-name", help="FiftyOne dataset name (default: <project>_v<version>)")
    parser.add_argument("--train-tag", default="train", help="Tag for training samples (default: train)")
    parser.add_argument("--val-tag", default="val", help="Tag for validation samples (default: val)")
    parser.add_argument("--test-tag", default="test", help="Tag for test samples (default: test)")
    
    # Interactive mode
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Enter interactive mode to be prompted for parameters")
    
    args = parser.parse_args(argv)
    
    # Ensure local config exists
    config_path = Path(args.config).expanduser() if args.config else None
    ensure_local_config(config_path=config_path)
    
    # Interactive mode or missing required params
    if args.interactive or not all([args.version, args.format]):
        args = get_interactive_params(args, config_path=config_path)
    
    # Get API key using priority system
    try:
        if not args.api_key:
            args.api_key = get_roboflow_api_key(config_path=config_path)
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    # Get workspace from config if not provided
    if not args.workspace:
        args.workspace = get_roboflow_default("workspace", config_path=config_path)
        if not args.workspace:
            print("❌ Error: Workspace ID is required. Provide via --workspace or config.")
            sys.exit(1)
    
    # Get project from config if not provided
    if not args.project:
        args.project = get_roboflow_default("project", config_path=config_path)
        if not args.project:
            print("❌ Error: Project ID is required. Provide via --project or config.")
            sys.exit(1)
    
    # Validate required params
    if not args.version:
        print("❌ Error: Dataset version is required. Provide via --version.")
        sys.exit(1)
    
    if not args.format:
        print("❌ Error: Dataset format is required. Provide via --format.")
        sys.exit(1)
    
    # Set download directory
    if not args.download_dir:
        data_dir = get_data_dir(config_path=config_path)
        args.download_dir = os.path.join(data_dir, "roboflow", args.project)
    
    # Set FiftyOne dataset name
    if args.import_fiftyone and not args.fiftyone_name:
        args.fiftyone_name = f"{args.project}_v{args.version}"
    
    # Print configuration
    print("\n" + "="*60)
    print("Roboflow Dataset Download")
    print("="*60)
    print(f"Workspace:       {args.workspace}")
    print(f"Project:         {args.project}")
    print(f"Version:         {args.version}")
    print(f"Format:          {args.format}")
    print(f"Download Dir:    {args.download_dir}")
    print(f"Unzip:           {args.unzip}")
    print(f"Delete Zip:      {args.delete_zip}")
    if args.import_fiftyone:
        print(f"Import FiftyOne: Yes")
        print(f"FO Dataset Name: {args.fiftyone_name}")
        print(f"Tags:            {args.train_tag}, {args.val_tag}, {args.test_tag}")
    else:
        print(f"Import FiftyOne: No")
    print("="*60 + "\n")
    
    try:
        # Download dataset
        dataset_path = download_dataset(
            api_key=args.api_key,
            workspace=args.workspace,
            project=args.project,
            version=args.version,
            format_type=args.format,
            download_dir=args.download_dir,
        )
        
        # Unzip if requested
        zip_path = None
        if args.unzip:
            zip_path = unzip_dataset(args.download_dir)
        
        # Delete zip if requested
        if args.delete_zip and zip_path and os.path.exists(zip_path):
            print(f"🗑️  Deleting zip file: {zip_path}")
            os.remove(zip_path)
        
        # Import to FiftyOne if requested
        if args.import_fiftyone:
            success = import_to_fiftyone(
                dataset_path=dataset_path,
                format_type=args.format,
                dataset_name=args.fiftyone_name,
                train_tag=args.train_tag,
                val_tag=args.val_tag,
                test_tag=args.test_tag,
            )
            
            if not success:
                print("\n⚠️  Download completed, but FiftyOne import was not successful.")
                print("   You can manually import the dataset later.")
                sys.exit(1)
        
        print("\n✅ All operations completed successfully!")
        print(f"   Dataset location: {dataset_path}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    raise SystemExit(main())
