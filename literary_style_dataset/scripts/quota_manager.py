import time
import json
from pathlib import Path

class QuotaManager:
    def __init__(self, config_path="configs/providers.json"):
        self.state = {}
        self.config_path = config_path
        self._load_config()

    def _load_config(self):
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        for provider, data in self.config.get("providers", {}).items():
            self.state[provider] = {
                "requests_used": 0,
                "tokens_used": 0,
                "last_request_time": 0,
                "cooldown_status": False,
                "cooldown_until": 0,
                "rpm_limit": data.get("rpm_limit", 10),
                "requests_in_current_minute": 0,
                "minute_start_time": time.time()
            }

    def can_make_request(self, provider):
        """Checks if a provider is available and not in cooldown."""
        state = self.state.get(provider)
        if not state:
            return False
            
        if state["cooldown_status"]:
            if time.time() > state["cooldown_until"]:
                state["cooldown_status"] = False
                state["requests_in_current_minute"] = 0
                state["minute_start_time"] = time.time()
            else:
                return False

        # Check RPM
        now = time.time()
        if now - state["minute_start_time"] > 60:
            state["minute_start_time"] = now
            state["requests_in_current_minute"] = 0
            
        if state["requests_in_current_minute"] >= state["rpm_limit"]:
            return False
            
        return True

    def record_request(self, provider, tokens_used=0):
        """Records a successful request."""
        state = self.state.get(provider)
        if state:
            state["requests_used"] += 1
            state["requests_in_current_minute"] += 1
            state["tokens_used"] += tokens_used
            state["last_request_time"] = time.time()

    def trigger_cooldown(self, provider, seconds=60):
        """Puts a provider in cooldown."""
        state = self.state.get(provider)
        if state:
            state["cooldown_status"] = True
            state["cooldown_until"] = time.time() + seconds
            print(f"[QuotaManager] {provider} triggered cooldown for {seconds}s.")

    def get_status(self):
        """Returns the current state of all providers."""
        return self.state
