from mingpt.model import GPT
from mingpt.bpe import BPETokenizer
import torch
from time import sleep
from json import load
import logging
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='[%H:%M:%S]', level=logging.INFO)

from train import CharDataset, get_config
from src.path_handler import model_file, stoi_file, itos_file

glob_cfg = get_config()
cfg = glob_cfg.model
cfg.vocab_size = 11056
cfg.block_size = 128

model = GPT(cfg)
model.load_state_dict(torch.load(model_file))

""" with open(data_file, 'r') as f:
    train_dataset = CharDataset(glob_cfg.data, f.read()) """

with open(stoi_file, 'r', encoding='utf-8') as f:
    stoi = load(f)
with open(itos_file, 'r', encoding='utf-8') as f:
    itos = load(f)

logging.info(f"Loaded dictionaries stoi and itos with sizes of {len(stoi)} and {len(itos)}")

def generate(prompt='', steps=1, do_sample=True):
    x = torch.tensor([stoi[tkn] for tkn in prompt.split(' ')], dtype=torch.long)[None,...].to('cpu')

    # forward the model `steps` times to get samples, in a batch
    y = model.generate(x, max_new_tokens=steps, do_sample=do_sample, top_k=40)
    
    all_tokens = [itos[str(int(i))] for i in y[0]]
    out = ' '.join(all_tokens)
    return out

def generate_tail(prompt='', steps=1, do_sample=True):
    x = torch.tensor([stoi[tkn] for tkn in prompt.split()], dtype=torch.long)[None,...].to('cpu')

    # forward the model `steps` times to get samples, in a batch
    y = model.generate(x, max_new_tokens=steps, do_sample=do_sample, top_k=40)
    
    all_tokens = [itos[str(int(i))] for i in y[0]]
    out = ' '.join(all_tokens[-steps:])
    return out

def generate_segment(context='-', stop_flag='\n \n'):
    text = context
    flag = False
    while not (flag and text.endswith(stop_flag)):
        flag = True
        text = generate(text)
    stripped_lines = [ x.removeprefix(' ') for x in text.split('\n') ]
    return '\n'.join(stripped_lines).strip()

def main():
    base = input(' >> ')
    print()
    steps = 1
    #total = 50

    print(base, end=' ', flush=True)

    while True:
        #total -= steps
        tail = generate_tail(base, steps=steps)
        print(tail, end=' ', flush=True)
        base += f' {tail}'
        sleep(0.01)

if __name__ == "__main__":
    #main()
    print(generate_segment())
