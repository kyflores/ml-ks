import transformers as tfs
import peft
import torch

import get_data
import make_model

SYSTEM_PROMPT = "Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> {thought with steps separated with '\n\n'} <|end_of_thought|> Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|> Now, try to solve the following question through the above guidelines:"

def generate_instruct(
    model,
    tokenizer,
    instruction,
    max_new_tokens=1024,
    temperature=0.5,
    top_k=50,
    repetition_penalty=1.1
):
    chat = [
        {"role": "system", "content": "{}".format(SYSTEM_PROMPT)},
        {"role": "user", "content": "{}".format(instruction)},
    ]
    text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    model = model.to(torch.bfloat16)
    model.eval()
    inputs = tokenizer(text, return_tensors='pt', truncation=True).to(model.device)
    print("Prompt has", len(inputs["input_ids"][0]), "tokens")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            synced_gpus=False,
        )

        output = tokenizer.batch_decode(output)[0]
        return output

def train(opt):
    rank = 128
    name = "HuggingFaceTB/SmolLM2-360M-Instruct"
    cfg, mdl, tok = make_model.load_model(name, max_len=4096)

    bespoke = get_data.get_bespoke(tok)

    lora_config = peft.LoraConfig(
        # This is the rank you see in all the LoRA materials
        r=rank,
        # These are (almost) all of the linear layers. You can experiment by training fewer of them.
        # We added new tokens so we must train the token embeddings.
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens"],
        task_type=peft.TaskType.CAUSAL_LM,
        # Rule of thumb for alpha seems to be 1-2x the rank.
        lora_alpha=1 * rank,
        lora_dropout=0.05
    )
    lora_model = peft.get_peft_model(mdl, lora_config)
    lora_model.print_trainable_parameters()

    # These 3 things are your main training parameters.
    lr=5e-5
    # Lower this if you get CUDA out of memory, but try to keep
    # (batchsize * gradient_accumulation_steps) at least 8.
    batchsize=2
    gradient_accumulation_steps=8
    epochs=3

    # Clear CUDA cache incase the user reruns this cell a bunch of times.
    torch.cuda.empty_cache()
    args = tfs.TrainingArguments(
        output_dir='./finetune',
        optim='adamw_torch',
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batchsize,
        per_device_eval_batch_size=batchsize,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=False,
        save_strategy="no",
        weight_decay=0.01,
        push_to_hub=False,
        report_to='none',
        torch_empty_cache_steps=100,
        bf16=True,
        use_cpu=False,
        tf32=True # Comment this if it gives you an error. It requires Ampere or newer.
    )

    collator = get_data.DataCollatorForOversizedSeq(tok)

    trainer = tfs.Trainer(
        model=lora_model,
        args=args,
        train_dataset=bespoke,
        processing_class=tok,
        data_collator=collator
    )

    trainer.train()
    lora_model.save_pretrained(save_directory="mlhi-lora-instruct", save_embedding_layers=True)

if __name__ == '__main__':
    train(None)
