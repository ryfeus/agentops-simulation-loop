FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt /opt/task/requirements.txt
RUN python -m pip install --no-cache-dir -r /opt/task/requirements.txt

COPY seed.sql /opt/task/seed.sql
RUN mkdir -p /app/data /logs/agent \
    && python -c "import pathlib,sqlite3; sql=pathlib.Path('/opt/task/seed.sql').read_text(); connection=sqlite3.connect('/app/data/billing.db'); connection.executescript(sql); connection.close()" \
    && chmod 0777 /app/data /logs/agent \
    && chmod 0666 /app/data/billing.db
