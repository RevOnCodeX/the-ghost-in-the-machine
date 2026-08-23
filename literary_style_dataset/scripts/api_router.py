import os
import json
import urllib.request
import urllib.error
from pathlib import Path
from dotenv import load_dotenv

from quota_manager import QuotaManager
from retry_manager import RetryManager

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

class APIRouter:
    def __init__(self):
        self.quota_manager = QuotaManager(BASE_DIR / "configs/providers.json")
        self.retry_manager = RetryManager(max_retries=2, base_delay=2)
        self._load_configs()
        self._check_credentials()

    def _load_configs(self):
        with open(BASE_DIR / "configs/providers.json", "r") as f:
            self.providers = json.load(f)["providers"]
        with open(BASE_DIR / "configs/models.json", "r") as f:
            self.models = json.load(f)["models"]

    def _check_credentials(self):
        keys = ["GEMINI_API_KEY", "OPENROUTER_API_KEY", "AGENTROUTE_API_KEY"]
        missing = [k for k in keys if not os.environ.get(k)]
        if missing:
            print(f"[WARNING] Missing API credentials in .env: {', '.join(missing)}")

    def _call_gemini(self, prompt, model_name):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key: raise Exception("GEMINI_API_KEY not found")
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key
        }
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"}
        }
        
        req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=30) as response:
            resp_body = json.loads(response.read().decode("utf-8"))
            content = resp_body["candidates"][0]["content"]["parts"][0]["text"]
            return content

    def _call_openrouter(self, prompt, model_name):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key: raise Exception("OPENROUTER_API_KEY not found")
        
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/RevOnCodeX/the-ghost-in-the-machine"
        }
        data = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=30) as response:
            resp_body = json.loads(response.read().decode("utf-8"))
            return resp_body["choices"][0]["message"]["content"]

    def _call_agentroute(self, prompt, model_name):
        api_key = os.environ.get("AGENTROUTE_API_KEY")
        if not api_key: raise Exception("AGENTROUTE_API_KEY not found")
        
        # Simulating AgentRoute endpoint as a standard OpenAI-compatible endpoint
        url = "https://api.agentroute.ai/v1/chat/completions" 
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        try:
            req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=30) as response:
                resp_body = json.loads(response.read().decode("utf-8"))
                return resp_body["choices"][0]["message"]["content"]
        except urllib.error.URLError:
            raise Exception("AgentRoute endpoint unreachable or mock URL.")

    def _make_request(self, provider_id, prompt):
        provider_config = self.providers.get(provider_id)
        model_name = provider_config["models"][0] if provider_config.get("models") else ""
        
        if provider_id == "gemini":
            return self._call_gemini(prompt, model_name), model_name
        elif provider_id == "openrouter":
            return self._call_openrouter(prompt, model_name), model_name
        elif provider_id == "agentroute":
            return self._call_agentroute(prompt, model_name), model_name
        else:
            raise Exception(f"Unsupported provider: {provider_id}")

    def route_request(self, prompt):
        """
        Routes the request to the highest priority available provider.
        Falls back to lower priority if a provider fails or hits limits.
        """
        sorted_providers = sorted(
            [p for p, data in self.providers.items() if data.get("enabled")], 
            key=lambda x: self.providers[x]["priority"]
        )
        
        for provider_id in sorted_providers:
            if not self.quota_manager.can_make_request(provider_id):
                continue
                
            # Attempt execution with retries
            def attempt_call():
                return self._make_request(provider_id, prompt)
                
            success, result = self.retry_manager.execute_with_retry(attempt_call)
            
            if success:
                content, model_name = result
                self.quota_manager.record_request(provider_id)
                return {
                    "success": True,
                    "content": content,
                    "provider": provider_id,
                    "model": model_name
                }
            else:
                # 429 or persistent failure, trigger cooldown and try next provider
                print(f"[APIRouter] Provider {provider_id} failed completely. Triggering cooldown.")
                self.quota_manager.trigger_cooldown(provider_id, seconds=120)
                continue
                
        return {
            "success": False,
            "error": "All providers exhausted."
        }
