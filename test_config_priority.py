
import os
import json
import sys
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.runtime_config import get_roboflow_api_key, get_roboflow_default, ensure_local_config, get_config_path

def test_priority():
    print("--- Testing Config Priority ---")
    
    # Setup: Ensure config exists and clean it
    config_path = get_config_path()
    ensure_local_config()
    
    original_config = {}
    if config_path.exists():
        with open(config_path, 'r') as f:
            original_config = json.load(f)
            
    try:
        # Case 1: Config has value, Env has different value -> Config wins
        print("1. Config > Env Var check...")
        test_config = {
            "roboflow": {
                "api_key": "config_key",
                "workspace": "config_workspace"
            }
        }
        with open(config_path, 'w') as f:
            json.dump(test_config, f)
            
        os.environ["ROBOFLOW_API_KEY"] = "env_key"
        os.environ["ROBOFLOW_WORKSPACE"] = "env_workspace"
        
        assert get_roboflow_api_key() == "config_key", f"Expected config_key, got {get_roboflow_api_key()}"
        assert get_roboflow_default("workspace") == "config_workspace", f"Expected config_workspace, got {get_roboflow_default('workspace')}"
        print("   PASS")
        
        # Case 2: Config missing/placeholder, Env has value -> Env wins
        print("2. Env Var Fallback check...")
        test_config_placeholder = {
            "roboflow": {
                "api_key": "<PLACEHOLDER>",
                "workspace": ""
            }
        }
        with open(config_path, 'w') as f:
            json.dump(test_config_placeholder, f)
            
        assert get_roboflow_api_key() == "env_key", f"Expected env_key, got {get_roboflow_api_key()}"
        assert get_roboflow_default("workspace") == "env_workspace", f"Expected env_workspace, got {get_roboflow_default('workspace')}"
        print("   PASS")

        # Case 3: Both missing -> Error or None
        print("3. Missing Validation check...")
        del os.environ["ROBOFLOW_API_KEY"]
        del os.environ["ROBOFLOW_WORKSPACE"]
        
        try:
            get_roboflow_api_key()
            print("   FAIL: Should have raised ValueError")
        except ValueError:
            print("   PASS: Value Error raised")
            
        assert get_roboflow_default("workspace") is None, "Should return None"
        print("   PASS: Default returned None")

    finally:
        # Restore original config
        if original_config:
            with open(config_path, 'w') as f:
                json.dump(original_config, f, indent=2)
        elif config_path.exists():
            config_path.unlink()

if __name__ == "__main__":
    test_priority()
