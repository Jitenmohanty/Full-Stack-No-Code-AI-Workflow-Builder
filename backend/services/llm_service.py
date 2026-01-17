import openai
import google.generativeai as genai
import httpx
import os
from typing import Dict, Optional

class LLMService:
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.serpapi_key = os.getenv("SERPAPI_KEY")
        self.brave_api_key = os.getenv("BRAVE_API_KEY")
    
    async def web_search(self, query: str, provider: str = "serpapi") -> str:
        """Perform web search"""
        if provider == "serpapi":
            url = "https://serpapi.com/search"
            params = {
                "q": query,
                "api_key": self.serpapi_key,
                "engine": "google"
            }
        else:  # brave
            url = "https://api.search.brave.com/res/v1/web/search"
            params = {"q": query}
            headers = {"X-Subscription-Token": self.brave_api_key}
        
        async with httpx.AsyncClient() as client:
            if provider == "serpapi":
                response = await client.get(url, params=params)
            else:
                response = await client.get(url, params=params, headers=headers)
            
            data = response.json()
            
            # Extract results
            if provider == "serpapi":
                results = data.get("organic_results", [])
                return "\n".join([f"{r.get('title')}: {r.get('snippet')}" for r in results[:3]])
            else:
                results = data.get("web", {}).get("results", [])
                return "\n".join([f"{r.get('title')}: {r.get('description')}" for r in results[:3]])
    
    async def generate_response(
        self,
        query: str,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        provider: str = "openai",
        model: str = "gpt-4",
        temperature: float = 0.7,
        enable_web_search: bool = False,
        search_provider: str = "serpapi"
    ) -> Dict:
        """Generate LLM response"""
        
        # Build prompt
        prompt = query
        if context:
            prompt = f"Context:\n{context}\n\nQuestion: {query}"
        
        # Add web search if enabled
        web_context = None
        if enable_web_search:
            web_context = await self.web_search(query, search_provider)
            prompt = f"Web Search Results:\n{web_context}\n\n{prompt}"
        
        # Generate response
        if provider == "openai":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = openai.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            
            return {
                "response": response.choices[0].message.content,
                "model": model,
                "web_context": web_context
            }
        
        else:  # gemini
            model_instance = genai.GenerativeModel(model)
            response = model_instance.generate_content(prompt)
            
            return {
                "response": response.text,
                "model": model,
                "web_context": web_context
            }