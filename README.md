# LLM Inference Docker

This project provides a Docker container for running inference with the Phi-3-mini language model using FastAPI.

## Prerequisites

- Docker
- NVIDIA GPU with CUDA support (optional, for GPU acceleration)

## Getting Started

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/llm-inference-docker.git
   cd llm-inference-docker
   ```

2. Build the Docker image:
   ```
   docker build -t llm-inference:latest .
   ```

3. Run the Docker container:

   For CPU:
   ```
   docker run -d -p 8000:8000 --name llm-inference-container llm-inference:latest
   ```

   For GPU (requires NVIDIA Docker):
   ```
   docker run -d -p 8000:8000 --gpus all --name llm-inference-container llm-inference:latest
   ```

4. The API will be available at `http://localhost:8000`

## API Usage

- GET `/`: Returns a welcome message
- POST `/inference`: Generates text based on input messages

Example curl command:
curl -X POST "http://localhost:8000/inference" \
-H "Content-Type: application/json" \
-d '{
"messages": [
{"role": "user", "content": "What are three interesting facts about the moon?"}
],
"max_new_tokens": 100
}'

## Configuration

You can configure the following environment variables:
- `PORT`: The port on which the FastAPI server will run (default: 8000)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
