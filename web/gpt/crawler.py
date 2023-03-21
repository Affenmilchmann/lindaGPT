from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from requests import get
from tqdm import tqdm

from src.path_handler import data_file

base_url = "https://www.anekdot.ru/an/an2302/j{date};100.html"
TEXT_LEN_LIMIT = 200
TOP_TRESHOLD = 10

delimeter = ""

input(f"Your file {data_file} will be overwritten if exists. Press enter to continue.")
with open(data_file, 'w') as f:
    pass

def date_generator(start="2001-01-01",end="2023-03-20", step=timedelta(days=1)):
    """both `start` and `end` are included"""
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    delta = end_dt - start_dt
    it: datetime = start_dt

    pb = tqdm(total=delta.days)

    while it <= end_dt:
        pb.set_description(f'{it.strftime("%d.%m.%Y")} / {end_dt.strftime("%d.%m.%Y")}')
        pb.update(1)
        yield it.strftime("%y%m%d")
        it += step

def parse_and_filter(parent_div: BeautifulSoup) -> bool:
    # only needed posts have such class
    if not 'data-id' in parent_div.attrs:
        return None
    # text class is the one with text :)
    try:
        text_div = parent_div.find('div', class_='text')
        text = text_div.get_text(separator='\n')
    except AttributeError as e:
        return None
    # filtering by rating 
    try:
        place = parent_div.find('div', class_='num').text
        if int(place) > TOP_TRESHOLD:
            return None
    except AttributeError as e:
        print(e)
    # too long ones will badly impact on result I guess 
    if len(text) > TEXT_LEN_LIMIT:
        return None
    return text

def parse_html(url: str):
    page = get(url)
    soup: BeautifulSoup = BeautifulSoup(page.text, 'html.parser')
    all_divs = soup.findAll('div', class_='topicbox')

    texts = [parse_and_filter(div) for div in all_divs]
    texts = filter(lambda x: x != None, texts)

    return list(texts)

def write_to_file(texts: list):
    with open(data_file, 'a', encoding='utf-8') as f:
        for text in texts:
            f.write(f"{delimeter}\n{text}\n")

for it in date_generator(start="2001-01-01"):
    write_to_file(parse_html(base_url.format(date=it)))
