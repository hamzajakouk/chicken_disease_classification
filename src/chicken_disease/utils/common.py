import os
from box.exceptions import BoxValueError
import yaml
from chicken_disease import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64
