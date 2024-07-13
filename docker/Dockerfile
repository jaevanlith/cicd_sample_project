# Use the official Python image from the Docker Hub
FROM python:3.10-alpine

# Set the working directory in the container
WORKDIR /code

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install build tools and dependencies
RUN apk update && \
    apk add --no-cache build-base gcc musl-dev linux-headers g++ && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Command to run the application (for example purposes)
CMD ["python", "app.py"]
