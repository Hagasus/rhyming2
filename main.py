from GPT import tokenizer, gpt, device

prompt = tokenizer("<|startoftext|>\nCategory: Love\nTags: Sad, Butterfly, Penis\nAuthor:", return_tensors='pt')
prompt = {key: value.to(device) for key, value in prompt.items()}
sample = gpt.generate(**prompt, min_length=150, max_length=150, do_sample=True)
print(tokenizer.decode(sample[0]))