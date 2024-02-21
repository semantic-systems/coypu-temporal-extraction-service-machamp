# Temporal Extraction Models

[![Coypu](https://pbs.twimg.com/profile_banners/1421069821723267072/1641392854/1500x500)](https://coypu.org/)
## Introduction and setting up


This repository is an implementation of the [forked repository based on the master's thesis "Extraction and Classification of Time in Unstructured Data" (2023)](https://github.com/skonline90/Temporal-Extraction). It dockerizes the projects and through an API allows the access and usage of these services.

The files for the container are in the ```build``` folder. The scripts of both services were merged there. If one needs separate files, the original repository contains them on folder names after each service.

In order to build and raise the container one must simply have [Docker](https://docs.docker.com/engine/install/) installed, position themselves on the root folder of the repository and run the following command:

```cmd
docker compose up --build -d
```

This will orchestrate the image and container, download the necessary files along with setting up the API.

## Endpoints
The call to both of these endpoints must be done with the ```POST``` standard of HTTP calls. The base URL that the call must be performed to is:
```url
https://temporal-extraction.skynet.coypu.org
```
For both type of services, the call expects a raw body in JSON format with the single value of  ```sentence```, like so:

```json
{"sentence": "I will go get bread tomorrow in the morning."}
```

The response of each of the services is different, however, they are both returned in a JSON format and encapsulated in the  ```results```  key.

### MACHAMP

To use the service of MACHAMP, one must add ```/machamp``` to the base URL when executing the call. Like so:

```
https://temporal-extraction.skynet.coypu.org/machamp
```

The result, using the body shown in the [Endpoints](#Endpoints) section, will have the following format:

```json
{
   "results":[
      {
         "token":"I",
         "value":"O"
      },
      {
         "token":"will",
         "value":"O"
      },
      {
         "token":"go",
         "value":"O"
      },
      {
         "token":"get",
         "value":"O"
      },
      {
         "token":"bread",
         "value":"O"
      },
      {
         "token":"tomorrow",
         "value":"B-DATE"
      },
      {
         "token":"in",
         "value":"O"
      },
      {
         "token":"the",
         "value":"O"
      },
      {
         "token":"morning",
         "value":"O"
      },
      {
         "token":".",
         "value":"O"
      }
   ]
}
```

### UIE

To use the service of UIE, one must add ```/uie``` to the base URL when executing the call. Like so:

```
https://temporal-extraction.skynet.coypu.org/uie
```

The result, using the body shown in the [Endpoints](#Endpoints) section, will have the following format:

```json
{
   "results":{
      "input_text":"I will go get bread tomorrow in the morning.",
      "record":{
         "entity":{
            "offset":[
               [
                  "date",
                  [
                     5
                  ]
               ],
               [
                  "time",
                  [
                     8
                  ]
               ]
            ],
            "string":[
               [
                  "date",
                  "tomorrow"
               ],
               [
                  "time",
                  "morning"
               ]
            ]
         },
         "event":{
            "offset":[
               
            ],
            "string":[
               
            ]
         },
         "relation":{
            "offset":[
               
            ],
            "string":[
               
            ]
         }
      },
      "seq2seq":"<extra_id_0><extra_id_0> date<extra_id_5> tomorrow<extra_id_1><extra_id_0> time<extra_id_5> morning<extra_id_1><extra_id_1>",
      "tagged_sentence":"I will go get bread <TIMEX3>tomorrow</TIMEX3> in the <TIMEX3>morning</TIMEX3>.",
      "tokens":[
         "I",
         "will",
         "go",
         "get",
         "bread",
         "tomorrow",
         "in",
         "the",
         "morning",
         "."
      ]
   }
}
```
