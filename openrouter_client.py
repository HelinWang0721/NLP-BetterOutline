import yaml
import time
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI


class OpenRouterClient:

    def __init__(self, config_path: str = "./config.yaml"):
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Get API key from config
        api_key = self.config['api']['api_key']
        
        if not api_key:
            raise ValueError(
                "OpenRouter API key is not set. Please add it to config.yaml:\n"
                "api:\n"
                "  api_key: \"sk-or-v1-your-key-here\""
            )
        
        # Initialize OpenAI client with OpenRouter endpoint
        base_url = self.config['api']['base_url']
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        self.total_cost = 0.0
        self.request_count = 0
        
        self.logger.info(f"OpenRouter client initialized with base URL: {base_url}")
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Send chat completion request to OpenRouter
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name (defaults to config)
            temperature: Sampling temperature (defaults to config)
            max_tokens: Maximum tokens to generate (defaults to config)
            top_p: Nucleus sampling parameter (defaults to config)
            **kwargs: Additional parameters
            
        Returns:
            Generated text response
        """
        # Use config defaults if not specified
        model = model or self.config['model']['name']
        temperature = temperature if temperature is not None else self.config['model']['temperature']
        max_tokens = max_tokens or self.config['model']['max_tokens']
        top_p = top_p if top_p is not None else self.config['model']['top_p']
        
        # Retry logic
        max_retries = kwargs.pop('max_retries', 3)
        retry_delay = kwargs.pop('retry_delay', 2)
        
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"Sending request to {model} (attempt {attempt + 1}/{max_retries})")
                
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stream=False,
                    **kwargs
                )
                
                # Track request
                self.request_count += 1
                
                # Extract response
                content = response.choices[0].message.content
                
                # Log token usage if available
                if hasattr(response, 'usage'):
                    usage = response.usage
                    self.logger.info(
                        f"Model: {model} | "
                        f"Prompt tokens: {usage.prompt_tokens} | "
                        f"Completion tokens: {usage.completion_tokens} | "
                        f"Total: {usage.total_tokens}"
                    )
                
                return content
                
            except Exception as e:
                self.logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:
                    # Wait before retrying
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    # Last attempt failed
                    self.logger.error(f"All {max_retries} attempts failed")
                    raise
    
    def chat_with_config(
        self,
        messages: List[Dict[str, str]],
        model_config: Dict[str, Any]
    ) -> str:
        """
        Chat using a model configuration dict
        
        Args:
            messages: List of message dicts
            model_config: Dict with 'model', 'temperature', 'max_tokens', etc.
            
        Returns:
            Generated text response
        """
        return self.chat(
            messages=messages,
            model=model_config.get('model'),
            temperature=model_config.get('temperature'),
            max_tokens=model_config.get('max_tokens'),
            top_p=model_config.get('top_p')
        )
    
    def count_tokens_estimate(self, text: str) -> int:
        """
        Estimate token count (rough approximation)
        More accurate counting would require tiktoken with specific model
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Estimated token count
        """
        # Rough estimate: ~4 characters per token for English, ~2 for Chinese
        # This is a simple heuristic
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        
        estimated_tokens = (chinese_chars // 2) + (other_chars // 4)
        return max(estimated_tokens, 1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            'total_requests': self.request_count,
            'estimated_total_cost': self.total_cost
        }


# Global instance (lazy initialization)
_client_instance = None

def get_client(config_path: str = "./config.yaml") -> OpenRouterClient:
    """Get or create the global OpenRouter client instance"""
    global _client_instance
    if _client_instance is None:
        _client_instance = OpenRouterClient(config_path)
    return _client_instance


def chat(
    messages: List[Dict[str, str]],
    **kwargs
) -> str:
    """
    Convenience function for chat
    Compatible with the old interface
    """
    client = get_client()
    return client.chat(messages, **kwargs)


if __name__ == "__main__":
    """Test the OpenRouter client"""
    import argparse
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="Test OpenRouter Client")
    parser.add_argument("--test", action="store_true", help="Run test request")
    parser.add_argument("--config", default="./config.yaml", help="Config file path")
    args = parser.parse_args()
    
    if args.test:
        print("Initializing OpenRouter client...")
        client = OpenRouterClient(args.config)
        
        print("\nRunning test request...")
        test_messages = [
            {"role": "user", "content": "你好，请用一句话介绍你自己。"}
        ]
        
        response = client.chat(test_messages)
        print(f"\nResponse: {response}")
        
        stats = client.get_stats()
        print(f"\nStats: {stats}")
        print("\nTest completed successfully!")
