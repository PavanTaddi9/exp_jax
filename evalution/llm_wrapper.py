import torch
from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig as TransformersGenerationConfig
from transformers import PreTrainedModel, PreTrainedTokenizer, StoppingCriteriaList


CODEGEMMA_MODEL_NAME = "google/codegemma-7b"
CODEGEMMA_CONTEXT_SIZE = 8192
CODEGEMMA_FIM_PREFIX = "<|fim_prefix|>"
CODEGEMMA_FIM_SUFFIX = "<|fim_suffix|>"
CODEGEMMA_FIM_MIDDLE = "<|fim_middle|>"
CODEGEMMA_STOP_TOKENS = ["<|file_separator|>", "<|fim_end|>", "<|end_of_turn|>"]


@dataclass(frozen=True)
class DecodingConfig:
    skip_special_tokens: bool = True

@dataclass(frozen=True)
class EncodingConfig:
    add_bos: bool = True
    add_eos: bool = False
    truncation: int | None = CODEGEMMA_CONTEXT_SIZE

@dataclass(frozen=True)
class TokenizationContext:
    tokenizer: PreTrainedTokenizer
    pad_token_id: int
    eos_token_id: int
    bos_token_id: int

    def encode(self, config: EncodingConfig, text_list: list[str]) -> list[list[int]]:
        inputs = self.tokenizer(
            text_list,
            add_special_tokens=False,
            truncation=config.truncation is not None,
            max_length=config.truncation,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return inputs["input_ids"]

    def decode(self, config: DecodingConfig, input_ids: torch.Tensor) -> list[str]:
        return self.tokenizer.batch_decode(
            input_ids, 
            skip_special_tokens=config.skip_special_tokens,
            clean_up_tokenization_spaces=False
        )

    def encode_with_padding(self, text_list: list[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            text_list,
            padding="longest",
            truncation=True,
            max_length=CODEGEMMA_CONTEXT_SIZE,
            return_tensors="pt",
            add_special_tokens=True,
        )
        return inputs.input_ids

@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 2048
    top_p: float = 0.95
    temperature: float = 0.7

    def to_transformers_config(self, eos_token_id: int) -> TransformersGenerationConfig:
        return TransformersGenerationConfig(
            max_new_tokens=self.max_new_tokens,
            top_p=self.top_p,
            temperature=self.temperature,
            eos_token_id=eos_token_id,
            pad_token_id=eos_token_id,  # CodeGemma doesn't have pad token
            do_sample=self.temperature > 0,
        )

@dataclass
class ModelContext:
    tokenization: TokenizationContext
    model: PreTrainedModel

    def generate(self, input_ids: torch.Tensor, config: GenerationConfig) -> torch.Tensor:
        input_len = input_ids.shape[1]
        if input_len >= CODEGEMMA_CONTEXT_SIZE:
            raise ValueError(f"Input length {input_len} exceeds context window")

        generation_config = config.to_transformers_config(
            self.tokenization.eos_token_id
        )

        outputs = self.model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            stopping_criteria=self._get_stopping_criteria(),
        )
        return outputs[:, input_len:]

    def _get_stopping_criteria(self):
        stop_ids = self.tokenization.tokenizer.convert_tokens_to_ids(CODEGEMMA_STOP_TOKENS)
        return StoppingCriteriaList([StopOnTokens(stop_ids)])


class StopOnTokens:
    def __init__(self, stop_token_ids: list[int]):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, _) -> bool:
        return any(token_id in self.stop_token_ids for token_id in input_ids[0])


def create_infilling_prompt(prefix: str, suffix: str) -> str:
    return f"{CODEGEMMA_FIM_PREFIX}{prefix}{CODEGEMMA_FIM_SUFFIX}{suffix}{CODEGEMMA_FIM_MIDDLE}"

def get_model_context() -> ModelContext:
    tokenizer = AutoTokenizer.from_pretrained(CODEGEMMA_MODEL_NAME)
    
    tokenization = TokenizationContext(
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id,  # CodeGemma uses EOS as pad
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )

    model = AutoModelForCausalLM.from_pretrained(
        CODEGEMMA_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    return ModelContext(tokenization, model)

