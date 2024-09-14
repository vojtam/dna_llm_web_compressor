
FROM python:3.12


WORKDIR /src


COPY ./requirements.txt .


RUN pip install --no-cache-dir --upgrade -r requirements.txt


COPY . .


CMD ["fastapi", "run", "src/app.py", "--port", "8000"]
