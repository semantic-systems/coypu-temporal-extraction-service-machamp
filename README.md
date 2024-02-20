# Temporal Extraction Models

## Introduction and setting up

[![Coypu](https://pbs.twimg.com/profile_banners/1421069821723267072/1641392854/1500x500)](https://coypu.org/)

This repository is an implementation of the [forked repository based on the master's thesis "Extraction and Classification of Time in Unstructured Data" (2023)](https://github.com/skonline90/Temporal-Extraction). It dockerizes the projects and through an API allows the access and usage of these services.

The files for the container are in the ```build``` folder. The scripts of both services were merged there. If one needs separate files, the original repository contains them on folder names after each service.

In order to build and raise the container one must simply have [Docker](https://docs.docker.com/engine/install/) installed, position themselves on the root folder of the repository and run the following command:

```
docker compose up --build -d
```

This will download the necessary files along with setting up the API.
