# Toxicity detector
A simple HTTP server for checking text toxicity (specifically for banned words).

# Model
This project uses the [one-for-all-toxicity-v3](https://huggingface.co/FredZhang7/one-for-all-toxicity-v3) model, which is distributed under the [CC-BY-4.0 license](https://choosealicense.com/licenses/cc-by-4.0).
The model supports multilingualism (55 languages). It was trained on the [toxi-text-3M dataset](https://huggingface.co/datasets/FredZhang7/toxi-text-3M).

# Installation
> [!WARNING]
> The project is designed to run on CPU, if you want to use GPU you will have to replace torch dependency in `pyproject.toml`.

> [!TIP]
> It works well on hetzner Shared vCPU AMD server with 4cpu, 8gb ram, handles messages under 100ms. Maybe less resources needed, check this out yourself.

## Local Run
> [!IMPORTANT]
>
> Minimum requirement [Python 3.9](https://www.python.org/downloads).
>
> This project uses [uv](https://astral.sh/uv) for dependency management, but it is also possible to install dependencies via pip. This is not necessary.

1. Clone the repository

    ```bash
    git clone https://github.com/twirapp/toxicity-detector.git && cd toxicity-detector
    ```
2. Download model
    Run this in the project root directory:
    ```bash
    mkdir -p ./model && cd ./model && \
    curl -O https://huggingface.co/FredZhang7/one-for-all-toxicity-v3/resolve/main/config.json && \
    curl -O https://huggingface.co/FredZhang7/one-for-all-toxicity-v3/resolve/main/pytorch_model.bin && \
    curl -O https://huggingface.co/FredZhang7/one-for-all-toxicity-v3/resolve/main/special_tokens_map.json && \
    curl -O https://huggingface.co/FredZhang7/one-for-all-toxicity-v3/resolve/main/tokenizer.json && \
    curl -O https://huggingface.co/FredZhang7/one-for-all-toxicity-v3/resolve/main/tokenizer_config.json && \
    curl -O https://huggingface.co/FredZhang7/one-for-all-toxicity-v3/resolve/main/vocab.txt && \
    cd ..
    ```
3. Install dependencies

    This will automatically create the virtual environment in the `.venv` directory and install the required dependencies
    ```bash
    uv sync
    ```
    <details>
    <summary>(not recommended) alternative install via pip</summary>
    Create a virtual environment and activate:

    ```bash
    python3 -m venv .venv && source .venv/bin/activate
    ```
    Install only the required dependencies:

    ```bash
    pip3 install --no-deps -r requirements.lock
    ```
    </details>
4. Run the server

    With autoload:
    ```bash
    uvicorn app.server:app --reload
    ```
    Without autoload:
    ```bash
    uvicorn app.server:app
    ```

## Docker Hub
You can pull the pre-built Docker image from Docker Hub:
```bash
docker pull twirapp/toxicity-detector
```

And run it with the command:
```
docker run --rm -p 8000:8000 --name toxicity-detector twirapp/toxicity-detector
```

## Docker Build
1. Clone the repository

  ```bash
  git clone https://github.com/twirapp/toxicity-detector.git && cd toxicity-detector
  ```
2. Build the Docker image

  ```bash
  docker build -t toxicity-detector .
  ```
3. Run the container

  ```bash
  docker run --rm -p 8000:8000 --name toxicity-detector toxicity-detector
  ```

## Docker Compose
Create a `docker-compose.yml` file with the following content:
```yml
services:
  toxicity-detector:
    image: twirapp/toxicity-detector
    ports:
      - "8000:8000"
    environment:
      TOXICITY_THRESHOLD: 0
      # WEB_CONCURRENCY: 1 # uvicorn workers count
```

Then run:
```bash
docker compose up -d
```

# Usage
Make a GET request to `/` or `/predict` (preferred) with query parameter `?text=your text here`.
The result will be `0` or `1`, `0` - the text is considered non-toxic, `1` - the text is considered toxic.
Curl command for testing:
```bash
curl -G 'http://localhost:8000/predict' --data-urlencode 'text=test text'
```
> [!NOTE]
> `--data-urlencode` is needed to work with letters other than English, for example Russian.

# Environment variables
- `MODEL_PATH` - path to the directory where the model files are stored. (which you should have downloaded) Default: `./model`
- `TOXICITY_THRESHOLD` - the level below which the text will be considered toxic. Default: `0` - the argmax function is used. This is a float value, example: `-0.2`, `-0.05`, `1`.
- `WEB_CONCURRENCY` - Number of worker processes. Defaults to the value of this environment variable if set, otherwise 1. Note: Not compatible with `--reload` option.

# Explanation of the log output
`01-24 19:01:34 | 0.568 sec |  9.583, -9.616 | False | 'text'`
1. `01-24 19:01:34` - date and time: month, day of the month, time of message display.
2. `0.568 sec` - execution time spent on the model call.
3. ` 9.583, -9.616` - returned value from the model. The first for how much toxic text, the second number the opposite of the first. When specifying `TOXICITY_THRESHOLD` you need to look at the first number. The more negative the first value, the more toxic the text.
4. `False` - prediction result based on `TOXICITY_THRESHOLD` (if set) or the result of the argmax function.
5. `'text'` - the text that was passed to the model. After clearing emoji and converting to lowercase.
