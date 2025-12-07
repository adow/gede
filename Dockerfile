FROM python:3.12.8-slim

COPY ./requirements.txt /codes/ 
COPY ./README.md /codes/

WORKDIR /codes
RUN pip3 install -r /codes/requirements.txt
COPY ./gede /codes/gede

CMD python3 -m gede.gede
