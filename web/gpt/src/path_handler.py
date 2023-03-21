from pathlib import Path

src_dir = Path(__file__).parent
gpt_dir = src_dir.parent
proj_dir = gpt_dir.parent
data_dir = gpt_dir.joinpath('data')

data_file = data_dir.joinpath('anekdots.txt')
model_file = gpt_dir.joinpath('out/chargpt/model.pt')
stoi_file = data_dir.joinpath('stoi.json')
itos_file = data_dir.joinpath('itos.json')
