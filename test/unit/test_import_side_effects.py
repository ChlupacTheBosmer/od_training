import os
import subprocess
import sys

import pytest


pytestmark = pytest.mark.unit


def test_importing_modules_does_not_create_local_config_file(tmp_path):
    """Importing runtime modules must not create config files as side effects."""
    cfg_path = tmp_path / "cfg" / "local_config.json"
    env = os.environ.copy()
    env["ODT_CONFIG_PATH"] = str(cfg_path)

    code = (
        "import od_training.dataset.manager;"
        "import od_training.train.yolo;"
        "import od_training.train.rfdetr;"
        "import od_training.infer.runner;"
        "import od_training.utility.verify_env"
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=env)

    assert not cfg_path.exists()

