import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LLM_Model:
    def __init__(self, model_id: str = "tiiuae/Falcon3-3B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_id
        )
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=False,
        ).to(self.device)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _get_model_inputs(self, base_prompt):
        dialogue_template = [{"role": "user", "content": base_prompt}]
        input_data = self.tokenizer.apply_chat_template(
            conversation=dialogue_template, tokenize=False, add_generation_prompt=True
        )
        input_data = self.tokenizer(input_data, return_tensors="pt").to(self.device)
        return input_data

    def run(self, base_prompt):
        input_data = self._get_model_inputs(base_prompt=base_prompt)
        output_ids = self.llm_model.generate(
            input_ids=input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            max_length=256,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response = response.split("<|assistant|>")[-1].strip()
        return response


if __name__ == "__main__":
    llm_model = LLM_Model()
    base_prompt = "Hello, how are you?"
    response = llm_model.run(base_prompt=base_prompt)
    print(response)
