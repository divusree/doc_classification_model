# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000/tcp
CMD [ "python", "-m" , "flask", "run", "--host=0.0.0.0", "--port=5000"]