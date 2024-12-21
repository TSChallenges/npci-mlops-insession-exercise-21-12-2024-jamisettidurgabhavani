# pull python base image
FROM python:3.10-slim

# add requirements file & trained model
ADD requirements.txt .

# update pip
RUN pip install --upgrade pip

# install dependencies
RUN pip install -r requirements.txt

RUN rm requirements.txt

ADD app/main.py app/.
# add application file
ADD app/model/*.pkl app/model/.

# expose port where your application will be running
EXPOSE 7860

# start application
CMD ["python", "app/main.py"]
