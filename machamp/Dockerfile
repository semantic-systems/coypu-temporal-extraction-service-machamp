FROM python:3.8-slim

COPY . /src/
WORKDIR src

RUN \
    apt-get update && \
    apt-get -y upgrade && \
    apt-get -y dist-upgrade && \
    apt-get -y autoremove && \
    apt-get -y install \
        git \
        curl \
        nano \
        build-essential \
        bash \
        wget \
        unzip \
    && \
    apt-get -y clean

RUN chmod +x download_finetuned_xlm-roberta_models.bash

RUN ./download_finetuned_xlm-roberta_models.bash

RUN pip install -r requirements.txt

RUN python ./download_punkt_tokenizer.py

EXPOSE 5555/tcp

ENTRYPOINT ["python", "/src/machamp_api.py"]
