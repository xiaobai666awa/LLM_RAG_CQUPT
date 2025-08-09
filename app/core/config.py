import json
from typing import Optional


class Config:
    def __init__(self, path: str = "../data/config.json"):
        self.path = path
        self._config = self._load_config()

    def _load_config(self) -> dict:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found at {self.path}")
        except json.JSONDecodeError:
            raise ValueError(f"Config file {self.path} contains invalid JSON.")

    def get_api_key(self, key_type: str) :
        print(key_type)
        return self._config.get("api_key").get(key_type)

    def get(self, key: str, default=None):
        return self._config.get(key, default)

    def check_api_keys(self) -> bool:
        api_keys = self._config.get("api_key", {})
        if not isinstance(api_keys, dict):
            return False
        return all(bool(value and str(value).strip()) for value in api_keys.values())

    def is_api_key_empty(self, key_type: str) -> bool:
        val = self.get_api_key(key_type)
        return not (val and str(val).strip())

    def update_api_key(self, key_type: str, new_value: str):
        if "api_key" not in self._config or not isinstance(self._config["api_key"], dict):
            self._config["api_key"] = {}

        self._config["api_key"][key_type] = new_value

        # 将修改写回 config.json
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._config, f, indent=2, ensure_ascii=False)

    def reload(self):
        """重新加载配置"""
        self._config = self._load_config()
