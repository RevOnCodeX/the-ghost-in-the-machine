import time

class RetryManager:
    def __init__(self, max_retries=2, base_delay=2):
        self.max_retries = max_retries
        self.base_delay = base_delay

    def execute_with_retry(self, func, *args, **kwargs):
        """
        Executes a function with retry logic.
        Retries up to max_retries times with exponential backoff.
        Returns (success_boolean, result_or_error_msg)
        """
        for attempt in range(self.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                return True, result
            except Exception as e:
                error_msg = str(e)
                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** attempt)
                    print(f"[RetryManager] Attempt {attempt + 1} failed: {error_msg}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    print(f"[RetryManager] All {self.max_retries} retries failed.")
                    return False, error_msg
