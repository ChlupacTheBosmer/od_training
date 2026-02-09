import argparse
from roboflow import Roboflow
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def upload_weights(api_key: str, workspace: str, project_name: str, version_num: int, weights_path: str, model_type: str = "yolov8"):
    """
    Uploads trained weights to Roboflow.
    """
    logger.info(f"Authenticating with Roboflow...")
    rf = Roboflow(api_key=api_key)
    
    logger.info(f"Accessing {workspace}/{project_name} v{version_num}...")
    project = rf.workspace(workspace).project(project_name)
    version = project.version(version_num)
    
    logger.info(f"Uploading weights from {weights_path} (Type: {model_type})...")
    try:
        # Note: 'deploy' is the method to upload weights for inference
        version.deploy(model_type=model_type, model_path=weights_path)
        logger.info("Upload successful.")
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload weights to Roboflow.")
    parser.add_argument("--api-key", required=True, help="Roboflow API Key (or set ROBOFLOW_API_KEY env var)")
    parser.add_argument("--workspace", required=True, help="Roboflow Workspace ID")
    parser.add_argument("--project", required=True, help="Roboflow Project ID")
    parser.add_argument("--version", type=int, required=True, help="Dataset Version Number")
    parser.add_argument("--weights", required=True, help="Path to weights file (.pt)")
    parser.add_argument("--type", default="yolov8", help="Model type (yolov8, yolov5, rf-detr, etc.)")
    
    args = parser.parse_args()
    
    upload_weights(args.api_key, args.workspace, args.project, args.version, args.weights, args.type)
