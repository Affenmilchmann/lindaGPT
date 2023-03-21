"""
Trains a character-level language model.
"""

import os
import sys
import logging
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='[%H:%M:%S]', level=logging.INFO)

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN

# -----------------------------------------------------------------------------

from src.tokenizer import tokenize_text

input_file = 'data/anekdots.txt'

# --------This part is not mine------------------------------------------------

def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/chargpt'

    # data
    C.data = CharDataset.get_default_config()

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-mini'

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 2e-4 # 5e-4 0.0005

    return C

# -------This part is mine---------------------------------------------------

class CharDataset(Dataset):
    """
    Emits batches of characters
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 128
        return C

    def __init__(self, config, data):
        self.config = config

        tokens = tokenize_text(data)
        
        uniq_tokens = sorted(list(set(tokens))) # chars = sorted(list(set(data)))
        data_size, vocab_size = len(tokens), len(uniq_tokens)
        print('data has %d tokens, %d unique.' % (data_size, vocab_size))

        self.stoi = { ch:i for i,ch in enumerate(uniq_tokens) }
        self.itos = { i:ch for i,ch in enumerate(uniq_tokens) }
        self.vocab_size = vocab_size
        self.data = tokens

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.data) - self.config.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.config.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    logging.info('starting...')

    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    # construct the training dataset
    text = open(input_file, 'r', encoding='utf-8').read() # don't worry we won't run out of file handles
    train_dataset = CharDataset(config.data, text)

    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)

    # construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset)

    # iteration callback
    def batch_end_callback(trainer):

        if trainer.iter_num % 100 == 0:
            logging.info(f"iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

        if trainer.iter_num % 500 == 0:
            # evaluate both the train and test score
            model.eval()
            with torch.no_grad():
                # sample from the model...
                context = "штирлиц"
                x = torch.tensor([train_dataset.stoi[s] for s in context.split()], dtype=torch.long)[None,...].to(trainer.device)
                y = model.generate(x, 25, temperature=1.0, do_sample=True, top_k=10)[0]
                completion = ' '.join([train_dataset.itos[int(i)] for i in y])
                print(completion)
            # save the latest model
            print("saving model")
            ckpt_path = os.path.join(config.system.work_dir, "model.pt")
            torch.save(model.state_dict(), ckpt_path)
            # revert model to training mode
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    # run the optimization
    trainer.run()
