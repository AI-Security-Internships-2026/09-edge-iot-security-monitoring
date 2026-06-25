FROM flwr/superexec:1.31.0

WORKDIR /app

COPY pyproject.toml .
COPY src/ ./src/

# Strip out unnecessary simulation layers and install lightweight Scikit-Learn
RUN sed -i 's/.*flwr\[simulation\].*//' pyproject.toml \
    && python -m pip install -U --no-cache-dir \
       numpy pandas scikit-learn

ENTRYPOINT ["flower-superexec"]