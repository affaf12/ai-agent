"""
API Tester Agent - Test and analyze APIs
"""
from typing import Dict, Any, Optional
from .llm import LLMClient
import requests
import json

class APIAgent:
    """Test and document APIs"""
    
    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient(task_type="general")
    
    def test_endpoint(self, url: str, method: str = "GET", headers: Dict = None, data: Dict = None) -> Dict[str, Any]:
        """Test API endpoint"""
        try:
            headers = headers or {}
            
            if method.upper() == "GET":
                resp = requests.get(url, headers=headers, timeout=10)
            elif method.upper() == "POST":
                resp = requests.post(url, json=data, headers=headers, timeout=10)
            elif method.upper() == "PUT":
                resp = requests.put(url, json=data, headers=headers, timeout=10)
            elif method.upper() == "DELETE":
                resp = requests.delete(url, headers=headers, timeout=10)
            else:
                return {"success": False, "error": f"Method {method} not supported"}
            
            # Try to parse JSON
            try:
                body = resp.json()
                body_str = json.dumps(body, indent=2)[:2000]
            except:
                body_str = resp.text[:2000]
            
            return {
                "success": True,
                "status": resp.status_code,
                "headers": dict(resp.headers),
                "body": body_str,
                "time_ms": int(resp.elapsed.total_seconds() * 1000),
                "url": url,
                "method": method
            }
        except Exception as e:
            return {"success": False, "error": str(e), "url": url}
    
    def analyze_response(self, response_data: Dict) -> Dict[str, Any]:
        """Analyze API response with LLM"""
        if not response_data.get("success"):
            return response_data
        
        prompt = f"""Analyze this API response:

URL: {response_data['url']}
Status: {response_data['status']}
Time: {response_data['time_ms']}ms
Body: {response_data['body'][:1000]}

Provide:
1. Is it successful?
2. What data structure is returned
3. Any issues or warnings
4. Suggested improvements
"""
        
        analysis = self.llm.chat(prompt)
        
        return {
            **response_data,
            "analysis": analysis
        }
    
    def generate_docs(self, base_url: str, endpoints: list) -> Dict[str, Any]:
        """Generate API documentation"""
        results = []
        for ep in endpoints:
            test = self.test_endpoint(
                f"{base_url}{ep['path']}", 
                ep.get('method', 'GET')
            )
            results.append(test)
        
        prompt = f"""Create API documentation from these test results:

{json.dumps(results, indent=2)}

Generate markdown documentation with endpoints, methods, responses."""
        
        docs = self.llm.chat(prompt)
        
        return {"success": True, "documentation": docs, "tests": results}
