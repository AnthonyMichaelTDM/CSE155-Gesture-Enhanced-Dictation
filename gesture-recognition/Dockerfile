FROM python:3.12.7

# set working directory
WORKDIR /app

# set any necessary environment variables

# ENV ENVIRONMENT production
# ENV LOG_LEVEL INFO

# install dependencies
RUN apt-get update && apt-get install -y \
    python3-opencv \
    pulseaudio \
    libgl1-mesa-glx \
    libxkbcommon-x11-0

# install requirements
# first copy the requirements file separately so we can cache the pip install step
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# install additional dependencies
# ...

# copy the rest of the application code
COPY app.py /app/app.py
COPY utils /app/utils
COPY model /app/model
COPY volume-mute.png /app/volume-mute.png
COPY Chime.mp3 /app/Chime.mp3

# run the application
CMD ["python", "app.py"]