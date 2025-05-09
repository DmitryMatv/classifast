# Use an official Python runtime as a parent image
FROM python:3.13-slim

# Set the working directory in the container
WORKDIR /service_root

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire local 'app' directory (which should contain main.py, classifier.py, static, templates, __init__.py)
# to a directory named 'app' inside the WORKDIR.
# Structure inside container: /service_root/app/main.py, /service_root/app/classifier.py etc.
COPY ./app ./app

# Make port 6009 available to the world outside this container
EXPOSE 6009

# Command to run the application using Uvicorn
# Uvicorn will look for 'app.main:app' relative to the WORKDIR /service_root
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "6009"]