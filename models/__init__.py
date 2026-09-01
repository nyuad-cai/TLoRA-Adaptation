#import other model handlers here 
from .mistral import MistralSmallHandler #first name is name of python file and second name is that of the class from mistral.py 
from .medgemma import MedGemma27BMCQHandler
from .meditron import Meditron3MCqHandler
from .med42 import Med42MCQHandler
from .deepseek import DeepSeekV32MCQHandler
from .fanar import Fanar19BMCQHandler
from .gemini import Gemini3ProHandler
from .claude import ClaudeOpus45MCQHandler
from .jais import Jais2ChatMCQHandler
from .falcon import FalconH1MCQHandler
from .silma import Silma9BMCQHandler
from .bimedix import BiMediXMCQHandler
from .autocap import AutoCAPMCQHandler

import os

def load_model_handler(config):
    """
    Load the appropriate model handler based on config
    """

    model_cfg = config["model"]
    model_type = model_cfg["type"]

    if model_type == "openai":
        api_key = model_cfg.get("api_key", os.getenv("OPENAI_API_KEY"))
        if not api_key:
            raise ValueError(
                "API key is not provided in config or environment variable (OPENAI_API_KEY)."
            )

        return OpenAIHandler(
            api_key=api_key,
            model=model_cfg["name"]
        )

    elif model_type == "mistral":
        return MistralSmallHandler(
            model_name=model_cfg.get(
                "name",
                "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
            ),
            cache_dir=model_cfg.get("cache_dir"),
            offline=model_cfg.get("offline", True),
        )

    elif model_type == "med42":
        return Med42MCQHandler(
            model_name=model_cfg.get(
            "name",
            "m42-health/Llama3-Med42-70B"
            ),
            cache_dir=model_cfg.get("cache_dir"),
            offline=model_cfg.get("offline", True),
            
        )

    elif model_type == "jais":
        return Jais2ChatMCQHandler(
            model_name=model_cfg.get("name", "inceptionai/Jais-2-8B-Chat"),
            cache_dir=model_cfg.get("cache_dir"),
            offline=model_cfg.get("offline", True),
        )

    elif model_type == "gemini":
        return Gemini3ProHandler(
            api_key=os.getenv("GEMINI_API_KEY"),
            model=model_cfg.get(
                "name",
                "gemini-3.1-pro-preview"
            ),
        )

    elif model_type == "fanar":
        return Fanar19BMCQHandler(
            model_name=model_cfg.get(
            "name",
            "QCRI/Fanar-1-9"
            ),
            cache_dir=model_cfg.get("cache_dir"),
            offline=model_cfg.get("offline", True),
            
        )

    elif model_type == "autocap":
        return AutoCAPMCQHandler(
            model_name=model_cfg.get(
                "name",
                "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
            ),
            cache_dir=model_cfg.get("cache_dir"),
            offline=model_cfg.get("offline", True),
            candidate_languages=model_cfg.get(
                "candidate_languages",
                ["Arabic", "English", "French"]
            ),
            top_k_languages=model_cfg.get("top_k_languages", 3),
            selection_max_tokens=model_cfg.get("selection_max_tokens", 96),
            weight_max_tokens=model_cfg.get("weight_max_tokens", 96),
            reasoning_max_tokens=model_cfg.get("reasoning_max_tokens", 32),
            do_sample=model_cfg.get("do_sample", False),
        )

    elif model_type == "bimedix":
        return BiMediXMCQHandler(
            model_name=model_cfg.get("name", "BiMediX/BiMediX-Bi"),
            cache_dir=model_cfg.get("cache_dir"),
            offline=model_cfg.get("offline", True),
        )
        
    elif model_type == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY") or model_cfg.get("api_key")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is not set (env var or config).")
    
        return DeepSeekV32MCQHandler(
            api_key=api_key,
            model_name=model_cfg.get("model_name", "deepseek-chat"),
            max_retries=model_cfg.get("max_retries", 3),
            # do NOT pass cache_dir/offline to API handlers
        )

    elif model_type == "claude":
        api_key = os.getenv("ANTHROPIC_API_KEY") or model_cfg.get("api_key")
        if not api_key:
            raise ValueError("ANTHROPI_API_KEY is not set (env var or config).")
    
        return ClaudeOpus45MCQHandler(
            api_key=api_key,
            model_name=model_cfg.get("model_name", "claude-opus-4-6"),
            max_retries=model_cfg.get("max_retries", 3),
            # do NOT pass cache_dir/offline to API handlers
        )

    elif model_type == "falcon":
        return FalconH1MCQHandler(
            model_name=model_cfg.get(
                "name",
                "tiiuae/Falcon-H1-7B-Instruct"
            ),
            cache_dir=model_cfg.get("cache_dir"),
            offline=model_cfg.get("offline", True),
        )

    elif model_type == "silma":
        return Silma9BMCQHandler(
            model_name=model_cfg.get("name", "silma-ai/SILMA-9B-Instruct-v1.0"),
            cache_dir=model_cfg.get("cache_dir"),
            offline=model_cfg.get("offline", True),
        )
    
    
    elif model_type == "meditron":
        return Meditron3MCqHandler(
            model_name=model_cfg.get(
            "name",
            "OpenMeditron/Meditron3-70B"
            ),
            cache_dir=model_cfg.get("cache_dir"),
            offline=model_cfg.get("offline", True),
        )

    elif model_type == "medgemma":
        return MedGemma27BMCQHandler(model_name=model_cfg.get(
            "name",
            "google/medgemma-27b-text-it"
        ),
        cache_dir=model_cfg.get("cache_dir"),
        offline=model_cfg.get("offline", True),
        )

    else:
        raise ValueError(f"Unsupported model type: {model_type}")
