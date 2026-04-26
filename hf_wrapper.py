import requests
import json

class DummyMessage:
    def __init__(self, content):
        self.content = content

class DummyChoice:
    def __init__(self, content):
        self.message = DummyMessage(content)

class DummyResponse:
    def __init__(self, content):
        self.choices = [DummyChoice(content)]

class HFWrapper:
    def __init__(self, base_url, api_key):
        # Strip /v1/ or /v1 from the URL because raw HF endpoints don't use it
        self.base_url = base_url.replace('/v1/', '').replace('/v1', '')
        if self.base_url.endswith('/'):
            self.base_url = self.base_url[:-1]
        self.api_key = api_key
        
        class Chat:
            def __init__(self, outer):
                self.outer = outer
                
            class Completions:
                def __init__(self, outer):
                    self.outer = outer
                    
                def create(self, model, messages, **kwargs):
                    # Convert OpenAI messages to a single string prompt
                    prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
                    
                    headers = {
                        "Authorization": f"Bearer {self.outer.outer.api_key}",
                        "Content-Type": "application/json"
                    }
                    
                    # Some endpoints require the new conversational format if they are slightly advanced,
                    # but since this is a raw Transformers container, we use `inputs`.
                    payload = {
                        "inputs": prompt,
                        "parameters": {
                            "max_new_tokens": kwargs.get("max_tokens", 150),
                            "temperature": kwargs.get("temperature", 0.1),
                            "return_full_text": False
                        }
                    }
                    
                    try:
                        r = requests.post(self.outer.outer.base_url, headers=headers, json=payload, timeout=120)
                        if r.status_code != 200:
                            raise Exception(f"HF API Error: {r.status_code} {r.text}")
                        
                        res = r.json()
                        # Raw text generation endpoints return a list with generated_text
                        if isinstance(res, list) and len(res) > 0 and "generated_text" in res[0]:
                            content = res[0]["generated_text"].strip()
                            return DummyResponse(content)
                        # Fallback parsing
                        return DummyResponse(str(res))
                    except Exception as e:
                        raise Exception(f"Wrapper error: {e}")
                        
            @property
            def completions(self):
                return self.Completions(self)
                
        self.chat = Chat(self)
