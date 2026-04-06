import os
import json
import yaml
import logging
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

load_dotenv(find_dotenv())
_client_cache: dict = {}

def load_config():
    with open("configs/settings.yaml", "r") as f:
        return yaml.safe_load(f)

def load_prompts():
    with open("configs/prompts.yaml", "r") as f:
        return yaml.safe_load(f)


def get_openai_client(config):
    key = config["llm"]["base_url"]
    if key not in _client_cache:
        _client_cache[key] = OpenAI(
            base_url=config["llm"]["base_url"],
            api_key=os.getenv(config["llm"]["api_key_env"]),
            timeout=30.0,   # hard timeout per request
            max_retries=3,  # retry on transient errors
        )
    return _client_cache[key]

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler("run.log"), logging.StreamHandler()]
    )

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)