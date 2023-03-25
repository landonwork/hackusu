
import torch
from transformers import GPT2Tokenizer
import torch.nn.functional as F


# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
tokenizer = GPT2Tokenizer.from_pretrained('./tokenizer/', map_location=device)
model = torch.load('shakespeare_translator', map_location=device)

def translate(
    prompt,
    entry_count=1,
    entry_length=30,
    top_p = 0.8,
    temperature=1.,
):
    if not isinstance(prompt, str):
        raise TypeError('prompt must be a str')
    if '<startoftext>' not in prompt and isinstance(prompt, str):
        prompt = '<startoftext> ' + prompt.lstrip()
    if '<transition>' not in prompt and isinstance(prompt, str):
        prompt = prompt.rstrip() + ' <transition>'

    model.eval()
    generated_num = 0
    generated_list = []

    filter_value = -float("Inf")

    with torch.no_grad():
        for entry_idx in range(entry_count):
            entry_finished = False
            generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
            for i in range(entry_length):
                outputs = model(generated, labels=generated)
                loss, logits = outputs[:2]
                logits = logits[:, -1, :]/(temperature if temperature > 0 else 1.0)

                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value

                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                generated = torch.cat((generated, next_token), dim=-1)

                if next_token in tokenizer.encode("<endoftext>"):
                    entry_finished = True

                if entry_finished:
                    generated_num = generated_num + 1

                    output_list = list(generated.squeeze().numpy())
                    output_text = tokenizer.decode(output_list)
                    generated_list.append(output_text)
                    break

            if not entry_finished:
                output_list = list(generated.squeeze().numpy())
                output_text = f"{tokenizer.decode(output_list)}<endoftext>"
                generated_list.append(output_text)

    return generated_list[0].split('<transition> ')[1].rstrip('<end')

import re

def clean(text):
  if '<end' not in text:
    text = '. '.join(text.split('.')[:-1])
  else:
    text = re.sub(r'<end(oftext>)?', '.', text)
  text = re.sub(r'[\[\]]', '', text)
  return text

def generate(
    prompt,
    entry_count=1,
    entry_length=50,
    top_p = 0.8,
    temperature=1.,
):

    model.eval()
    generated_num=0
    generated_list=[]

    filter_value = -float("Inf")

    with torch.no_grad():
        for entry_idx in range(entry_count):
            entry_finished = False
            generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
            for i in range(entry_length):
                outputs = model(generated, labels=generated)
                loss, logits = outputs[:2]
                logits = logits[:, -1, :]/(temperature if temperature > 0 else 1.0)

                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value

                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                generated = torch.cat((generated, next_token), dim=-1)

                if next_token in tokenizer.encode("<endoftext>"):
                    entry_finished = True

                if entry_finished:
                    generated_num = generated_num + 1

                    output_list = list(generated.squeeze().numpy())
                    output_text = tokenizer.decode(output_list)
                    generated_list.append(output_text)
                    break

            if not entry_finished:
                output_list = list(generated.squeeze().numpy())
                output_text = f"{tokenizer.decode(output_list)}<endoftext>"
                generated_list.append(output_text)

    return clean(generated_list[0])

