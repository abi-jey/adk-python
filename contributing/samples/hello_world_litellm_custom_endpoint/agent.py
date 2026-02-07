# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sample agent using LiteLlm with a custom model endpoint.

This example shows how to point ADK to an external or third-party model
provider by setting the ``api_base`` parameter on the ``LiteLlm`` wrapper.
Any OpenAI-compatible server (e.g. vLLM, Ollama, LocalAI, or your own
deployment) can be used as the backend.

Before running, set the environment variables for your provider:
  export CUSTOM_MODEL_API_BASE="https://my-server.example.com/v1"
  export CUSTOM_MODEL_API_KEY="my-api-key"         # if required
  export CUSTOM_MODEL_NAME="openai/my-model-name"  # LiteLLM model string
"""

import os
import random

from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm


def roll_die(sides: int) -> int:
  """Roll a die and return the rolled result.

  Args:
    sides: The integer number of sides the die has.

  Returns:
    An integer of the result of rolling the die.
  """
  return random.randint(1, sides)


# Read configuration from environment so secrets stay out of source code.
_API_BASE = os.environ.get(
    "CUSTOM_MODEL_API_BASE", "http://localhost:8000/v1"
)
_API_KEY = os.environ.get("CUSTOM_MODEL_API_KEY", "")
_MODEL_NAME = os.environ.get("CUSTOM_MODEL_NAME", "openai/my-model")

root_agent = Agent(
    model=LiteLlm(
        model=_MODEL_NAME,
        api_base=_API_BASE,
        api_key=_API_KEY,
    ),
    name="custom_endpoint_agent",
    description=(
        "A simple agent that uses an external model provider via a custom"
        " endpoint."
    ),
    instruction="""
      You roll dice and answer questions about the outcome of the dice rolls.
      When you are asked to roll a die, you must call the roll_die tool with
      the number of sides. Be sure to pass in an integer.
      You should never roll a die on your own.
    """,
    tools=[
        roll_die,
    ],
)
