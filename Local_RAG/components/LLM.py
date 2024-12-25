import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LLM_Model:
    """
    A class to create and use a large language model (LLM) for generating text responses.

    This class initializes the language model, tokenizer, and prepares the model for inference.
    It can be used to generate text responses based on given prompts.

    Attributes:
        device (str): The device to run the language model on ("cuda" if GPU is available, else "cpu").
        model_id (str): The identifier for the pre-trained language model to use.
        tokenizer (AutoTokenizer): Tokenizer for encoding and decoding text inputs and outputs.
        llm_model (AutoModelForCausalLM): The language model for generating text.
    """

    def __init__(self, model_id: str = "tiiuae/Falcon3-3B-Instruct"):
        """
        Initializes the LLM_Model class with the specified language model.

        Args:
            model_id (str): Identifier for the pre-trained language model (default: "tiiuae/Falcon3-3B-Instruct").
        """
        # Set the device based on the availability of a GPU.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id

        # Load the tokenizer for the specified model.
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_id
        )

        # Load the language model with the specified configuration.
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=False,
        ).to(self.device)

        # Set the pad token ID to the end-of-sequence (EOS) token if it is not already set.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _get_model_inputs(self, base_prompt):
        """
        Prepares the input data for the language model.

        Args:
            base_prompt (str): The base prompt or input text for the model.

        Returns:
            dict: A dictionary containing the input_ids and attention_mask tensors for the model.
        """
        # Define a dialogue template with the user's role and content.
        dialogue_template = [{"role": "user", "content": base_prompt}]

        # Use the tokenizer to apply the chat template to the input prompt.
        input_data = self.tokenizer.apply_chat_template(
            conversation=dialogue_template, tokenize=False, add_generation_prompt=True
        )

        # Convert the dialogue into input tensors suitable for the model.
        input_data = self.tokenizer(input_data, return_tensors="pt").to(self.device)
        return input_data

    def run(self, base_prompt):
        """
        Executes the text generation process based on the provided base prompt.

        Args:
            base_prompt (str): The input prompt for which a response is to be generated.

        Returns:
            str: The generated text response from the language model.
        """
        # Get the model inputs from the base prompt.
        input_data = self._get_model_inputs(base_prompt=base_prompt)

        # Generate the text output from the model.
        output_ids = self.llm_model.generate(
            input_ids=input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            max_length=256,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Decode the generated output_ids to get the text response.
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Split the response to remove any extra content added by the model.
        response = response.split("<|assistant|>")[-1].strip()
        return response
