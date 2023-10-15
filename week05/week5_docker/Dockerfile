FROM svizor/zoomcamp-model:3.10.12-slim

#RUN apt update -y && apt install -y python3.9

#RUN rm /usr/local/bin/python && ln -s /usr/bin/python3.9 /usr/local/bin/python

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]

#Since we do not want to create virtual env but install system wide and deploy it we use the following
RUN pipenv install --deploy --system && \
    rm -rf /root/.cache

#copy my prediction app
COPY ["app.py", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "app:app"]