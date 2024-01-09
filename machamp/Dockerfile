FROM python:3.8-slim

RUN \
    apt-get update && \
    apt-get -y upgrade && \
    apt-get -y dist-upgrade && \
    apt-get -y autoremove && \
    apt-get -y install \
        git \
        curl \
        nano \
    && \
    apt-get -y clean

COPY . /src/
WORKDIR src

RUN pip install -r requirements.txt
CMD [ "python", "download_punkt_tokenizer.py"]
RUN /bin/bash -c 'download_finetuned_xlm-roberta_models.bash’'

EXPOSE 5292/tcp

ENTRYPOINT ["python"]
CMD ["/src/machamp_api.py"]
