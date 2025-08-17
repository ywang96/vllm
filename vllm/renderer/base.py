from typing import Optional, Any, TYPE_CHECKING
from pydantic import NotRequired
from vllm.sequence import ChatCompletionMessageParam
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.config import ModelConfig
from transformers import AutoTokenizer

if TYPE_CHECKING:
    from vllm.multimodal.inputs import MultiModalDataDict

class BaseRenderer:
    def __init__(self, model_config: ModelConfig):
        # initialize with model-specific tokenizer and processor, and any object required by the model
        # defined by the model vendor.
        self.tokenizer = AutoTokenizer.from_pretrained(model_config.model)

    def render_conversation(
               self, 
               convo: list[ChatCompletionMessageParam]
               ) -> tuple[list[int], Optional[list[MultiModalFeatureSpec]]]:
        """
        Convert an OpenAI-style request (chat or completion) into prompt token ids and optionally 
        multimodal features with metadata.
        """

        # Model vendor & developer has full control on how to define this conversion logic, but typically this
        # looks like:
        # 1. Convert list of messages to a single string-format prompt
        # 2. Tokenize the prompt to prompt token ids
        # 3. (Optional) Fetch multimodal media contents and process them into features (as inputs of multimodal encoder) 
        #  with metadata, and process input token ids to expand placeholder tokens. 
        prompt = self.tokenizer.apply_chat_template(convo, tokenize=False)
        return self.render_prompt(prompt)

    def render_prompt(
              self, 
              prompt: str, 
              multi_modal_data: NotRequired["MultiModalDataDict"], 
              mm_processor_kwargs: NotRequired[dict[str, Any]]
              ) -> tuple[list[int], Optional[list[MultiModalFeatureSpec]]]:
        """
        Used for `.generate` endpoint, and can be used inside `render_conversation` if defined.
        """ 
        return self.tokenizer.encode(prompt, return_tensors="pt")