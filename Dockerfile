FROM python:3.7

RUN mkdir /bot

COPY . /bot

RUN cd /bot && pip3 install -r requirements.txt

WORKDIR /bot

CMD ["python","main.py"]