FROM python:3.12.7

WORKDIR /app

# set an environment variable
ENV ENVIRONMENT production
ENV LOG_LEVEL INFO


# install dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg\ 
    libasound-dev\
    libportaudio2\
    libportaudiocpp0\
    portaudio19-dev\
    gcc


# Copy the requirements file separately so we can cache the pip install step
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# install additional dependencies
RUN apt-get install -y pulseaudio

# Copy the rest of the application code
COPY main.py /app/main.py
# COPY lib /app/lib

# Copy test audio file
COPY harvard.wav /app/harvard.wav

# Run the application
CMD ["python", "main.py"]