# Using a Custom / External Model Endpoint with ADK

This sample shows how to point ADK to **any** OpenAI-compatible model server
(e.g. vLLM, Ollama, LocalAI, a managed proxy, or your own deployment) using the
`LiteLlm` wrapper.

## Quick start

1. Set the environment variables for your provider:

```bash
export CUSTOM_MODEL_API_BASE="https://my-server.example.com/v1"
export CUSTOM_MODEL_API_KEY="my-api-key"         # if required by your server
export CUSTOM_MODEL_NAME="openai/my-model-name"  # LiteLLM model string
```

2. Launch the agent:

```bash
adk web contributing/samples
```

Then select **hello_world_litellm_custom_endpoint** in the UI.

## How it works

The `LiteLlm` wrapper accepts an `api_base` parameter that overrides the
default API endpoint used by LiteLLM. This lets you route requests to any
server that exposes an OpenAI-compatible chat completions API.

```python
from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm

root_agent = Agent(
    model=LiteLlm(
        model="openai/my-model",
        api_base="https://my-server.example.com/v1",
        api_key="my-api-key",
    ),
    name="my_agent",
    instruction="You are a helpful assistant.",
    tools=[...],
)
```

### Common provider examples

| Provider | `model` string | `api_base` |
|----------|---------------|------------|
| [vLLM](https://docs.vllm.ai/) | `openai/<your-model>` | `http://localhost:8000/v1` |
| [Ollama](https://ollama.com/) | `ollama_chat/<model-name>` | `http://localhost:11434` |
| [LocalAI](https://localai.io/) | `openai/<model-name>` | `http://localhost:8080/v1` |
| Any OpenAI-compatible proxy | `openai/<model-name>` | Your proxy URL |

For Ollama specifically, see the dedicated
[hello_world_ollama](../hello_world_ollama) sample for more details.

## Additional parameters

`LiteLlm` also forwards any extra keyword arguments to the underlying
[LiteLLM completion API](https://docs.litellm.ai/docs/completion/input),
so you can pass `api_version`, `custom_llm_provider`, `drop_params`,
`fallbacks`, and more.
