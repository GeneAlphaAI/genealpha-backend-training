from huggingface_hub import HfApi, create_repo, upload_file
from typing import Optional, Dict, Any
import os
from pathlib import Path

class HuggingFaceClient:
    """Handles HuggingFace Hub interactions"""
    
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("HUGGINGFACE_TOKEN")
        self.api = HfApi(token=self.token)
    
    def create_model_repo(self, repo_name: str, 
                         private: bool = False,
                         exist_ok: bool = True) -> str:
        """Create a model repository on HuggingFace Hub"""
        try:
            repo_url = create_repo(
                repo_id=repo_name,
                token=self.token,
                private=private,
                repo_type="model",
                exist_ok=exist_ok
            )
            return repo_url
        except Exception as e:
            raise ValueError(f"Failed to create repo: {str(e)}")
    
    def upload_model(self, 
                    model_path: str,
                    repo_id: str,
                    commit_message: str = "Upload model",
                    model_card: Optional[Dict] = None) -> str:
        """Upload model to HuggingFace Hub"""
        try:
            # Upload model file
            upload_file(
                path_or_fileobj=model_path,
                path_in_repo="model.pkl",
                repo_id=repo_id,
                token=self.token,
                commit_message=commit_message
            )
            
            # Create and upload model card if provided
            if model_card:
                model_card_path = self._create_model_card(model_card)
                upload_file(
                    path_or_fileobj=model_card_path,
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    token=self.token,
                    commit_message="Add model card"
                )
                os.remove(model_card_path)
            
            return f"https://huggingface.co/{repo_id}"
        except Exception as e:
            raise ValueError(f"Failed to upload model: {str(e)}")
    
    def _create_model_card(self, model_info: Dict) -> str:
        """Create a model card markdown file"""
        model_card = f"""
# {model_info.get('model_type', 'Model')}

## Model Description
- **Model Type**: {model_info.get('model_type', 'Unknown')}
- **Training Date**: {model_info.get('trained_at', 'Unknown')}
- **Job ID**: {model_info.get('job_id', 'Unknown')}

## Performance Metrics
"""
        
        if 'metrics' in model_info:
            for metric, value in model_info['metrics'].items():
                model_card += f"- **{metric}**: {value:.4f}\n"
        
        model_card += """
## Training Configuration
"""
        
        if 'config' in model_info:
            for param, value in model_info['config'].items():
                model_card += f"- **{param}**: {value}\n"
        
        model_card += """
## Usage

```python
import joblib

# Load the model
model = joblib.load('model.pkl')

# Make predictions
predictions = model.predict(X)
```
"""
        
        # Save to temporary file
        temp_path = "temp_model_card.md"
        with open(temp_path, 'w') as f:
            f.write(model_card)
        
        return temp_path
    
    def download_model(self, repo_id: str, 
                      local_dir: Optional[str] = None) -> str:
        """Download model from HuggingFace Hub"""
        from huggingface_hub import snapshot_download
        
        local_dir = local_dir or f"./downloaded_models/{repo_id}"
        
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                token=self.token
            )
            return local_dir
        except Exception as e:
            raise ValueError(f"Failed to download model: {str(e)}")
