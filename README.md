# News

In this repo, I decided to go beyond notebooks and implement a dockerized Api of the NewsCategorization project described 
[here](https://github.com/chimaobi-okite/NLP-Projects-Competitions/blob/main/NewsCategorization) while following mlops and software engineering best practices.

The project involves
- packaging the sklearn model development process following object-oriented programming best practices
- serving the model using Fastapi
- reproduciability with docker

Further steps would have been 
- integrating a CI/CD pipeline with github actions for continuous deployment of the api endpoint on heroku
- building a streamlit frontend that consumes the api. 
I did not include those for reasons related to already having a [gradio](https://huggingface.co/spaces/okite97/news-demo) frontend that showcases the model capabilities and cost.