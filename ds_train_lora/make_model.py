import transformers as tfs

def load_model(name: str, max_len=1024):
    config = tfs.AutoConfig.from_pretrained(name)
    model = tfs.AutoModelForCausalLM.from_pretrained(name)
    tokenizer = tfs.AutoTokenizer.from_pretrained(
        name,
        model_max_length=max_len
    )

    # TODO Required for PEFT to use gradient checkpointing https://github.com/huggingface/peft/issues/137
    # model.enable_input_require_grads()

    # chatml - requires <|im_start|> and <|im_end|> special tokens.
    # If they don't exist, tokenizer.add_special_tokens and model.resize_token_embeddings can be used, but
    # these tokens would come with randomly initialized embeddings and need finetuning.
    # Standard LoRA does not train input embeddings so this probably won't work without full fine tune.
    # See for details on chat template https://huggingface.co/docs/transformers/main/chat_templating#what-template-should-i-use
    # tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    # Mistral template uses no special tokens
    tokenizer.chat_template = "{%- for message in messages %}{%- if message['role'] == 'system' -%}{{- message['content'] -}}{%- else -%}{%- if message['role'] == 'user' -%}{{-'[INST] ' + message['content'].rstrip() + ' [/INST]'-}}{%- else -%}{{-'' + message['content'] + '</s>' -}}{%- endif -%}{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}{{-''-}}{%- endif -%}"
    # Using eos as the pad token seems common practice.
    tokenizer.pad_token = tokenizer.eos_token

    return config, model, tokenizer
