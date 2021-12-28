FROM python
# Define variables.

# Install Python requirements.
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python