from transformers import AutoModelForCausalLM, GemmaTokenizer, PreTrainedTokenizer
from transformers.generation import GenerationConfig as TransformersGenerationConfig
import torch

# ====================
# Configuration Classes
# ====================

@dataclass
class GenerationConfig:
    max_new_tokens: int = 2048
    top_p: float = 0.95
    temperature: float = 0.2
    max_length: int = 8192  # Matches CodeGemma's context size

    def to_transformers_generation_config(self, eos_token_id: int, pad_token_id: int):
        return TransformersGenerationConfig(
            max_new_tokens=self.max_new_tokens,
            top_p=self.top_p,
            temperature=self.temperature,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            do_sample=self.temperature > 0,
        )

@dataclass
class TokenizationContext:
    tokenizer: PreTrainedTokenizer
    pad_token: str
    eos_token: str
    bos_token: str

    def encode_with_padding(self, direction: str, prompts: list[str]) -> torch.Tensor:
        return self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt",
            add_special_tokens=True,
        ).input_ids

    def decode(self, output_ids: torch.Tensor) -> list[str]:
        return self.tokenizer.batch_decode(
            output_ids, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

# ================
# Core Functionality
# ================

def create_infilling_prompt(prefix: str, suffix: str) -> str:
    return f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"

def get_codegemma_context(
    model_name: str = "google/codegemma-7b",
    use_flash_attention: bool = True,
) -> tuple[TokenizationContext, AutoModelForCausalLM]:
    # Initialize tokenizer
    tokenizer = GemmaTokenizer.from_pretrained(model_name)
    
    # Configure tokenization context
    tokenization_context = TokenizationContext(
        tokenizer=tokenizer,
        pad_token=tokenizer.pad_token or "<pad>",
        eos_token=tokenizer.eos_token or "<end_of_turn>",
        bos_token=tokenizer.bos_token or "<bos>",
    )

    # Configure model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2" if use_flash_attention else "eager",
    )

    return tokenization_context, model

# ================
# Generation Class
# ================

class CodeGemmaGenerator:
    def __init__(self, model_name: str = "google/codegemma-7b"):
        self.tokenization, self.model = get_codegemma_context(model_name)
        self.device = self.model.device
        self.default_stop_tokens = ["<|file_separator|>", "<|fim_end|>", "<|end_of_turn|>"]

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        stop_tokens: list[str] | None = None,
    ) -> str:
        # Set up configuration
        config = config or GenerationConfig()
        
        # Encode input
        input_ids = self.tokenization.encode_with_padding([prompt]).to(self.device)
        
        # Calculate available context
        max_new_tokens = min(
            config.max_new_tokens,
            self.model.config.max_position_embeddings - input_ids.shape[1]
        )
        
        # Generate output
        outputs = self.model.generate(
            input_ids=input_ids,
            generation_config=config.to_transformers_generation_config(
                eos_token_id=self.tokenization.tokenizer.eos_token_id,
                pad_token_id=self.tokenization.tokenizer.pad_token_id
            ),
            stopping_criteria=self._get_stopping_criteria(stop_tokens),
            max_new_tokens=max_new_tokens,
        )

        # Decode and clean output
        return self._clean_output(
            self.tokenization.decode(outputs[:, input_ids.shape[1]:])[0]
        )

    def _get_stopping_criteria(self, stop_tokens: list[str] | None):
        if stop_tokens is None:
            stop_tokens = self.default_stop_tokens
            
        stop_ids = [
            self.tokenization.tokenizer.convert_tokens_to_ids(tok)
            for tok in stop_tokens
        ]
        return StoppingCriteriaList([StopOnTokens(stop_ids)])

    def _clean_output(self, text: str) -> str:
        # Remove any remaining special tokens
        for tok in self.default_stop_tokens + [self.tokenization.eos_token]:
            text = text.replace(tok, "")
        return text.strip()

# ================
# Helper Classes
# ================

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids: list[int]):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return any(tok in self.stop_token_ids for tok in input_ids[0])

