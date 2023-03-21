FROM python:3.10

WORKDIR /lindaGPT

COPY requirements.txt requirements.txt
COPY . .

RUN pip3 install -r requirements.txt

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0" ]
