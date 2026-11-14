# running triton server
docker run --gpus all --rm   -p 8000:8000   -p 8001:8001   -p 8002:8002   -v ${PWD}/model_repository:/models   nvcr.io/nvidia/tritonserver:24.01-py3   tritonserver --model-repository=/models

# running fastapi server
uv run uvicorn src.api.main:app --port 8080

# running the load tester
uv run python -m src.api.load_tester