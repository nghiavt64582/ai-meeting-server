import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from typing import Optional
from service.logger_setup import logger
import requests
import os
from huggingface_hub import HfFolder

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
        # if device is running on Windows, cache the hugging face in D:/.cache
        if os.name == "nt":
            print("set cache in D")
            os.environ["TRANSFORMERS_CACHE"] = "D:/.cache/huggingface"
            os.environ["HF_HOME"] = "D:/.cache/huggingface"
        self.model_id = model_id
        # Set the torch device. "cuda" for GPU if available, otherwise "cpu".
        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize model and tokenizer to None; they will be loaded later
        self.tokenizer = None
        self.model = None
        self.load_model(model_id)
        self.gemini_api_key = open("../.env").read().strip().split("=")[1]
        logger.info(f"Gemini API key loaded: {self.gemini_api_key[:5]}")
        logger.info(f"[{time.time()}] AiModel instance initialized. Call .load_model() to load the model.")

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

        logger.info(f"[{time.time()}] Loading tokenizer for {self.model_id}...")
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
        
        # Get the model's maximum input length (for BART-large-cnn, it's typically 1024)
        max_model_input_length = self.tokenizer.model_max_length
        logger.info(f"Model max input length: {max_model_input_length}")

        # Set padding side to 'left' for efficient generation
        self.tokenizer.padding_side = "left"

        logger.info(f"[{time.time()}] Loading model {self.model_id} to {self.device}...")

        start_time = time.time()

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.float32,
            trust_remote_code=True,
            device_map="auto"
        ).to(self.device)
        logger.info(f"Model {self.model_id} loaded successfully. Total time {time.time() - start_time:.2f} seconds.")

        if need_resize_embeddings:
            resize_start_time = time.time()
            logger.info(f"[{time.time()}] Resizing token embeddings to {len(self.tokenizer)}...")
            self.model.resize_token_embeddings(len(self.tokenizer))
            logger.info(f"Resized token embeddings in {time.time() - resize_start_time:.2f} seconds.")

        self.model.config.use_cache = False
        self.model.eval()

        if not getattr(self.tokenizer, "chat_template", None):
            self.tokenizer.chat_template = (
                "{% for m in messages %}"
                "{% if m['role'] == 'system' %}<|im_start|>system\n{{ m['content'] }}<|im_end|>\n"
                "{% elif m['role'] == 'user' %}<|im_start|>user\n{{ m['content'] }}<|im_end|>\n"
                "{% elif m['role'] == 'assistant' %}<|im_start|>assistant\n{{ m['content'] }}<|im_end|>\n"
                "{% endif %}"
                "{% endfor %}<|im_start|>assistant\n"
            )

    def preload_model(self, model_id: str):
        try:
            logger.info(f"\nLoading tokenizer for: {model_id}...")
            start_time = time.time()
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            logger.info(f"Successfully loaded tokenizer for {model_id}. Total time : {time.time() - start_time:.2f} seconds.")
        except Exception as e:
            logger.info(f"ERROR: Could not load tokenizer for {model_id}. Reason: {e}")
            logger.info("Please check if the model name is correct and you have an internet connection if not cached locally.")

    def generate_response(self, text: str, n_tokens: Optional[int] = 100) -> str:

        start = time.time()
        logger.info(f"Generating response for: {text}... with max tokens: {n_tokens}")
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
        # use cache if model is not Qwen 7B
        use_cache = self.model_id != "Qwen/Qwen-7B-Chat"
        logger.info(f"Generate by model {self.model_id}, use_cache: {use_cache}")
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=n_tokens, # Limit the number of new tokens generated
                do_sample=True,     # Enable sampling for more varied responses
                use_cache=use_cache,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        new_tokens = output_ids[0][input_ids.shape[-1]:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        logger.info(f"Response generated in {time.time() - start:.2f} seconds.")

        return answer
    
    def generate_response_by_gemini_api(self, text: str) -> str:
        logger.info(f"Generating response using Gemini API for: {text}...")
        gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        res = requests.post(gemini_url, json={
            "contents":[
                {
                    "parts":[
                        {
                            "text":text
                        }
                    ]
                }
            ]
        }, headers={"X-goog-api-key": self.gemini_api_key})
        if res.status_code == 200:
            json = res.json()
            data = json["candidates"][0]["content"]["parts"][0]["text"]
            return data
        else:
            logger.error(f"Error occurred: {res.status_code}, {res.text}")
            return {"error": "Failed to generate response"}

    def summarize(
        self, 
        text: str, 
        summary_prompt: str = "Summarize this conversation in a few sentences :", 
        n_tokens: Optional[int] = 1000,
        is_use_gemini: bool = False
    ) -> str:
        """
        Generate a summary of the provided text using the AI model.
        """
        prompt = f"{summary_prompt}\n{text}"
        logger.info(f"Summarizing text with prompt: {summary_prompt}, content {text}, n_tokens: {n_tokens}")
        if is_use_gemini:
            return self.generate_response_by_gemini_api(prompt)
        else:
            return self.generate_response(prompt, n_tokens=n_tokens)
        
    def get_cur_model(self) -> str:
        return self.model_id

ai_model = AiModel(model_id="Qwen/Qwen2-1.5B-Instruct", device="cpu")

# --- Example Usage (for standalone testing) ---
if __name__ == "__main__" and False:
    logger.info("Starting AiModel demonstration...")

    model_handler = AiModel(model_id="Qwen/Qwen2-1.5B-Instruct", device="cpu") # Default for CPU

    logger.info("\n--- Generating first response ---")
    test_question_1 = "What is the capital of France?"
    response_1 = model_handler.generate_response(test_question_1)
    logger.info(f"\nUser: {test_question_1}")
    logger.info(f"AI: {response_1}")

    logger.info("\n--- Generating second response ---")
    test_question_2 = "Tell me a short story about a brave knight."
    response_2 = model_handler.generate_response(test_question_2)
    logger.info(f"\nUser: {test_question_2}")
    logger.info(f"AI: {response_2}")

    logger.info("\nDemonstration complete.")
