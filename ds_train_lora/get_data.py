import datasets as dts
import transformers as tfs
import random


# tok is captured from the global namespace
# x keys are instruction, context, response, category
# Not all samples have a context, so we'll ignore it.
def dolly_chat(tok, x):
    chat = [
        # {"role": "system", "content": "{}".format(x["context"])},
        {"role": "user", "content": "{}".format(x["instruction"])},
        {"role": "assistant", "content": "{}".format(x["response"])},
    ]
    chat_formatted = tok.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True,
    )

    # TODO Padding to max length always seems to result in static VRAM usage, but
    # is slower on average since many samples are much shorter than max_length.
    # Want to debug why peak VRAM fluctuates a lot when length can vary, as this sometimes
    # OOMs midway through training.
    tokenized = tok(chat_formatted, padding="max_length", truncation=True)

    return {"text": chat_formatted, "input_ids": tokenized["input_ids"]}


def get_dolly(tok):
    def mapfn(x):
        return dolly_chat(tok, x)

    dolly = dts.load_dataset("databricks/databricks-dolly-15k")
    return dolly["train"].map(mapfn)


def bespoke_chat(tok, x):
    chat = [
        {"role": "system", "content": "{}".format(x["system"])},
        {"role": "user", "content": "{}".format(x["conversations"][0]['value'])},
        {"role": "assistant", "content": "{}".format(x["conversations"][1]['value'])},
    ]
    chat_formatted = tok.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True,
    )


    # TODO Padding to max length always seems to result in static VRAM usage, but
    # is slower on average since many samples are much shorter than max_length.
    # Want to debug why peak VRAM fluctuates a lot when length can vary, as this sometimes
    # OOMs midway through training.
    tokenized = tok(chat_formatted, padding="max_length", truncation=False)

    # print("Dataset", len(tokenized['input_ids']))
    return {"text": chat_formatted, "input_ids": tokenized["input_ids"]}


def get_bespoke(tok):
    def mapfn(x):
        return bespoke_chat(tok, x)
    # Instruct dataset. A possible alternative is "tatsu-lab/alpaca"
    bespoke = dts.load_dataset("bespokelabs/Bespoke-Stratos-17k")

    return bespoke["train"].map(mapfn, num_proc=8)

# This collator selects a random slice of oversized sequences equal to the
# size of the context length. This is implemented in the collator so that the
# slice is different on every observation of the data.
class DataCollatorForOversizedSeq(tfs.DataCollatorForLanguageModeling):
    def __init__(self, tok):
        super().__init__(tokenizer=tok, mlm=False)
        self.tok = tok
        self.max_len = tok.model_max_length

    def __call__(self, features, return_tensors=None):
        for f in features:
            if len(f['input_ids']) <= self.max_len:
                continue

            # print("Long Seq", len(f['input_ids']))
            extra = len(f['input_ids']) - self.max_len
            offset = random.randint(0, extra)
            f['input_ids'] = f['input_ids'][offset : offset + self.max_len]

        return super().__call__(features, return_tensors)
