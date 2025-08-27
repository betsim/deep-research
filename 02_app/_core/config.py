from pathlib import Path
import os
import yaml

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


class Config:
    _instance = None
    _config = None
    _env_loaded = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            self._load_config()

    def _load_config(self):
        # .../02_app
        app_dir = Path(__file__).resolve().parents[1]
        config_path = app_dir / "config_app.yaml"
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML syntax in configuration file: {e}")

        # Resolve .env relative to the config file location
        dotenv_rel = self._config.get("api_keys", {}).get("dotenv_path", ".env")
        env_path = (config_path.parent / dotenv_rel).resolve()

        # Load environment variables exactly from resolved .env
        if not self._env_loaded:
            if load_dotenv:
                load_dotenv(dotenv_path=env_path, override=False)
            else:
                if env_path.exists():
                    for line in env_path.read_text(encoding="utf-8").splitlines():
                        s = line.strip()
                        if s and not s.startswith("#") and "=" in s:
                            k, v = s.split("=", 1)
                            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            self._env_loaded = True

        # Validate required env vars (defaults to OPENROUTER_API_KEY)
        required = self._config.get("api_keys", {}).get("required", ["OPENROUTER_API_KEY"])
        missing = [k for k in required if not os.getenv(k)]
        if missing:
            raise RuntimeError(
                f"Missing required environment variables: {', '.join(missing)}. "
                f"Expected in {env_path}"
            )

    def __getitem__(self, key):
        return self._config[key]

    def get(self, key, default=None):
        return self._config.get(key, default)


config = Config()
