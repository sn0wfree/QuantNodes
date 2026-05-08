"""
Strategy Service - Strategy validation and management
"""

from typing import Optional, Dict, Any


class StrategyService:
    """Strategy service for validation and management"""

    def __init__(self):
        pass

    async def validate_yaml(self, yaml_content: str) -> Dict[str, Any]:
        """Validate strategy YAML configuration"""
        try:
            import yaml
        except ImportError:
            # Fallback if yaml is not installed
            if not yaml_content.strip():
                return {"valid": False, "error": "Empty YAML content"}
            if "\t" in yaml_content:
                return {"valid": False, "error": "YAML cannot contain tabs"}
            return {"valid": True}

        try:
            parsed = yaml.safe_load(yaml_content)
            
            # Basic structure validation
            if not isinstance(parsed, dict):
                return {"valid": False, "error": "YAML must be a mapping"}
            
            # Check for required sections
            if "strategy" not in parsed:
                return {"valid": False, "error": "Missing 'strategy' section"}
            
            strategy = parsed.get("strategy", {})
            if not strategy.get("name"):
                return {"valid": False, "error": "Strategy name is required"}
            
            # Check for signals
            if "signals" not in parsed:
                return {"valid": False, "error": "Missing 'signals' section"}
            
            signals = parsed.get("signals", [])
            if not isinstance(signals, list) or len(signals) == 0:
                return {"valid": False, "error": "At least one signal is required"}
            
            for i, signal in enumerate(signals):
                if not signal.get("name"):
                    return {"valid": False, "error": f"Signal {i+1} is missing a name"}
                if not signal.get("formula"):
                    return {"valid": False, "error": f"Signal '{signal.get('name')}' is missing a formula"}
            
            # Check for portfolio section
            if "portfolio" not in parsed:
                return {"valid": False, "error": "Missing 'portfolio' section"}
            
            return {"valid": True}
            
        except yaml.YAMLError as e:
            return {"valid": False, "error": f"YAML syntax error: {str(e)}"}
        except Exception as e:
            return {"valid": False, "error": str(e)}

    async def parse_strategy(self, yaml_content: str) -> Optional[Dict[str, Any]]:
        """Parse strategy YAML into structured format"""
        try:
            import yaml
            parsed = yaml.safe_load(yaml_content)
            return parsed
        except Exception:
            return None


# Singleton instance
strategy_service = StrategyService()
