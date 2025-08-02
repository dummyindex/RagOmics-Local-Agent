"""Standalone function block loader without dependencies on other modules."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define minimal models needed
class Arg:
    def __init__(self, **kwargs):
        self.name = kwargs.get('name')
        self.value_type = kwargs.get('value_type')
        self.description = kwargs.get('description')
        self.optional = kwargs.get('optional', False)
        self.default_value = kwargs.get('default_value')


class StaticConfig:
    def __init__(self, **kwargs):
        self.args = [Arg(**arg) for arg in kwargs.get('args', [])]
        self.description = kwargs.get('description', '')
        self.tag = kwargs.get('tag', '')
        self.input_specification = kwargs.get('input_specification')
        self.output_specification = kwargs.get('output_specification')


class FunctionBlock:
    def __init__(self, **kwargs):
        self.name = kwargs.get('name')
        self.type = kwargs.get('type')
        self.description = kwargs.get('description')
        self.static_config = kwargs.get('static_config')
        self.code = kwargs.get('code', '')
        self.requirements = kwargs.get('requirements', '')
        self.parameters = kwargs.get('parameters', {})


class FunctionBlockLoaderStandalone:
    """Load function blocks from directory structure without external dependencies."""
    
    def __init__(self, base_dir: Union[str, Path]):
        """Initialize loader with base directory."""
        self.base_dir = Path(base_dir)
        if not self.base_dir.exists():
            raise ValueError(f"Base directory does not exist: {self.base_dir}")
    
    def load_function_block(self, block_path: Union[str, Path]) -> Optional[FunctionBlock]:
        """Load a single function block from directory."""
        block_dir = Path(block_path)
        if not block_dir.is_absolute():
            block_dir = self.base_dir / block_dir
        
        if not block_dir.exists():
            logger.error(f"Function block directory not found: {block_dir}")
            return None
        
        # Required files
        config_file = block_dir / "config.json"
        code_file = block_dir / "code.py"  # or code.R
        
        if not config_file.exists():
            logger.error(f"Config file not found: {config_file}")
            return None
        
        # Load config
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return None
        
        # Determine code file
        if not code_file.exists():
            code_file = block_dir / "code.R"
            if not code_file.exists():
                logger.error(f"Code file not found in: {block_dir}")
                return None
        
        # Load code
        try:
            with open(code_file, 'r') as f:
                code = f.read()
        except Exception as e:
            logger.error(f"Error loading code: {e}")
            return None
        
        # Load requirements if exists
        requirements = ""
        req_file = block_dir / "requirements.txt"
        if req_file.exists():
            try:
                with open(req_file, 'r') as f:
                    requirements = f.read()
            except Exception as e:
                logger.warning(f"Error loading requirements: {e}")
        
        # Build function block
        try:
            static_config_data = config_data.get('static_config', {})
            static_config = StaticConfig(**static_config_data)
            
            fb = FunctionBlock(
                name=config_data['name'],
                type=config_data['type'],
                description=config_data['description'],
                static_config=static_config,
                code=code,
                requirements=requirements
            )
            
            logger.info(f"Loaded function block: {fb.name}")
            return fb
            
        except Exception as e:
            logger.error(f"Error creating function block: {e}")
            return None
    
    def list_available_blocks(self) -> List[Dict[str, str]]:
        """List all available function blocks."""
        blocks = []
        
        # Search for config.json files
        for config_file in self.base_dir.rglob("config.json"):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                block_info = {
                    "name": config.get("name", "unknown"),
                    "type": config.get("type", "unknown"),
                    "description": config.get("description", ""),
                    "path": str(config_file.parent.relative_to(self.base_dir)),
                    "tags": config.get("tags", [])
                }
                blocks.append(block_info)
                
            except Exception as e:
                logger.warning(f"Error reading config {config_file}: {e}")
                continue
        
        return blocks