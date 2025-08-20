import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from typing import Optional

class AiModel:
    """
    A class to encapsulate the Qwen language model and tokenizer for text generation.
    Handles model loading, tokenizer configuration, and response generation.
    """
    def __init__(self, model_id: str = "Qwen/Qwen2-1.5B-Instruct", device: str = "cpu"):
        """
        Initializes the AiModel with a specified model ID and device.
        Note: The actual model and tokenizer loading is deferred to the 'load' method.

        Args:
            model_id (str): The Hugging Face model ID (e.g., "Qwen/Qwen2-1.5B-Instruct").
            device (str): The device to potentially load the model on ("cpu" or "cuda").
        """
        self.model_id = model_id
        # Set the torch device. "cuda" for GPU if available, otherwise "cpu".
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Initialize model and tokenizer to None; they will be loaded later
        self.tokenizer = None
        self.model = None
        self.load_model(model_id)
        print(f"[{time.time()}] AiModel instance initialized. Call .load_model() to load the model.")

    def load_model(self, model_id: str = None):
        """
        Loads the tokenizer and the language model.
        This method should be called after the AiModel object is initialized.

        Args:
            model_id (str, optional): The Hugging Face model ID to load.
                                      If None, uses the model_id provided during __init__.
        """
        if model_id:
            self.model_id = model_id
        
        if not self.model_id:
            raise ValueError("Model ID not specified. Please provide it during initialization or in the load method.")

        print(f"[{time.time()}] Loading tokenizer for {self.model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)

        need_resize_embeddings = False
        # Handle missing EOS and PAD tokens for Qwen models as per common practice
        if self.tokenizer.eos_token is None and self.tokenizer.pad_token is None:
            # Add a single special token and point both EOS and PAD to it
            special = {"additional_special_tokens": ["<|im_end|>"]}
            self.tokenizer.add_special_tokens(special)
            # Now assign both to the same string so they resolve to the SAME id
            self.tokenizer.eos_token = "<|im_end|>"
            self.tokenizer.pad_token = "<|im_end|>"
            need_resize_embeddings = True
        elif self.tokenizer.eos_token is None:
            # Reuse PAD as EOS if PAD exists; otherwise add one
            if self.tokenizer.pad_token is not None:
                self.tokenizer.eos_token = self.tokenizer.pad_token
            else:
                self.tokenizer.add_special_tokens({"additional_special_tokens": ["<|im_end|>"]})
                self.tokenizer.eos_token = "<|im_end|>"
                self.tokenizer.pad_token = "<|im_end|>"
                need_resize_embeddings = True
        elif self.tokenizer.pad_token is None:
            # If EOS exists, reuse it for PAD
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Set padding side to 'left' for efficient generation
        self.tokenizer.padding_side = "left"

        print(f"[{time.time()}] Loading model {self.model_id} to {self.device}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32, # Using float32 as specified in original code
            trust_remote_code=True,
        ).to(self.device) # Move the model to the determined device
        
        # Resize token embeddings if new special tokens were added
        if need_resize_embeddings:
            print(f"[{time.time()}] Resizing token embeddings to {len(self.tokenizer)}...")
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.config.use_cache = False # Keep use_cache False as in your original setup
        self.model.eval() # Set model to evaluation mode

        # Supply a minimal ChatML template if missing for consistent conversation formatting
        if not getattr(self.tokenizer, "chat_template", None):
            self.tokenizer.chat_template = (
                "{% for m in messages %}"
                "{% if m['role'] == 'system' %}<|im_start|>system\n{{ m['content'] }}<|im_end|>\n"
                "{% elif m['role'] == 'user' %}<|im_start|>user\n{{ m['content'] }}<|im_end|>\n"
                "{% elif m['role'] == 'assistant' %}<|im_start|>assistant\n{{ m['content'] }}<|im_end|>\n"
                "{% endif %}"
                "{% endfor %}<|im_start|>assistant\n"
            )

    def generate_response(self, text: str, n_tokens: Optional[int] = 100) -> str:
        """
        Generates a text response from the Qwen model based on the input text.

        Args:
            text (str): The input text (user's question/query).

        Returns:
            str: The generated response from the AI model.

        Raises:
            RuntimeError: If tokenizer padding ID is missing or chat template fails.
        """

        start = time.time()
        print(f"Generating response for: {text}")
        messages = [
            {"role": "user", "content": f"{text.strip()}"}
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048, # Use max_length to prevent excessively long inputs
        )
        if not torch.is_tensor(input_ids):
            raise RuntimeError(f"apply_chat_template did not return a tensor, got: {type(input_ids)}")

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise RuntimeError("tokenizer.pad_token_id is None; make sure you set tokenizer.pad_token and/or eos_token, then resize embeddings if you added tokens.")
        attention_mask = (input_ids != pad_id).long()

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=n_tokens, # Limit the number of new tokens generated
                do_sample=True,     # Enable sampling for more varied responses
                use_cache=True,    # As per your original configuration (Note: often True for speed)
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        new_tokens = output_ids[0][input_ids.shape[-1]:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        print(f"Response generated in {time.time() - start:.2f} seconds.")

        return answer

main_model = AiModel(model_id="Qwen/Qwen2-1.5B-Instruct", device="cpu")

# --- Example Usage (for standalone testing) ---
if __name__ == "__main__" and False:
    print("Starting AiModel demonstration...")

    model_handler = AiModel(model_id="Qwen/Qwen2-1.5B-Instruct", device="cpu") # Default for CPU

    print("\n--- Generating first response ---")
    test_question_1 = "What is the capital of France?"
    response_1 = model_handler.generate_response(test_question_1)
    print(f"\nUser: {test_question_1}")
    print(f"AI: {response_1}")

    print("\n--- Generating second response ---")
    test_question_2 = "Tell me a short story about a brave knight."
    response_2 = model_handler.generate_response(test_question_2)
    print(f"\nUser: {test_question_2}")
    print(f"AI: {response_2}")

    print("\nDemonstration complete.")
