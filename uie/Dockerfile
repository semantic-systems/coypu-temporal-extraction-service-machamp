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


RUN wget https://www.fdr.uni-hamburg.de/record/13621/files/large_fullpate_multi.zip
RUN unzip large_fullpate_multi.zip -d finetuned_models/
RUN rm large_fullpate_multi.zip

RUN pip install -r requirements.txt

RUN python ./download_punkt_tokenizer.py

EXPOSE 5001/tcp

ENTRYPOINT ["python", "/src/ui_api.py"]
