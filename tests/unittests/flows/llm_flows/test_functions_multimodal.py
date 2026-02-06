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

"""Tests for multimodal function tool results.

Verifies that FunctionTools can natively return types.Part or list[types.Part]
and have those parts included alongside the function response in the content
sent to the LLM.
"""

from google.adk.agents.llm_agent import Agent
from google.genai import types

from ... import testing_utils


def test_function_returning_single_part():
  """Test that a function returning a single types.Part includes it in content."""
  function_call = types.Part.from_function_call(name='get_image', args={})
  image_part = types.Part.from_bytes(data=b'\x89PNG\r\n', mime_type='image/png')

  responses = [
      function_call,
      'The image shows a cat.',
  ]
  mock_model = testing_utils.MockModel.create(responses=responses)

  def get_image() -> types.Part:
    """Returns an image."""
    return image_part

  agent = Agent(name='root_agent', model=mock_model, tools=[get_image])
  runner = testing_utils.InMemoryRunner(agent)
  events = runner.run('describe the image')

  # The function response event should contain both the function response
  # part and the image part.
  function_response_event = events[1]
  assert len(function_response_event.content.parts) == 2
  assert function_response_event.content.parts[0].function_response
  assert function_response_event.content.parts[1].inline_data

  # Verify the LLM request also received both parts.
  llm_request_contents = mock_model.requests[1].contents
  last_user_content = llm_request_contents[-1]
  assert last_user_content.role == 'user'
  assert len(last_user_content.parts) == 2
  assert last_user_content.parts[0].function_response
  assert last_user_content.parts[1].inline_data


def test_function_returning_list_of_parts():
  """Test that a function returning list[types.Part] includes all parts."""
  function_call = types.Part.from_function_call(
      name='get_image_with_caption', args={}
  )
  image_part = types.Part.from_bytes(data=b'\x89PNG\r\n', mime_type='image/png')
  text_part = types.Part.from_text(text='A photo of a sunset.')

  responses = [
      function_call,
      'Beautiful sunset!',
  ]
  mock_model = testing_utils.MockModel.create(responses=responses)

  def get_image_with_caption() -> list[types.Part]:
    """Returns an image with caption."""
    return [image_part, text_part]

  agent = Agent(
      name='root_agent', model=mock_model, tools=[get_image_with_caption]
  )
  runner = testing_utils.InMemoryRunner(agent)
  events = runner.run('describe the image')

  # The function response event should contain the function response part
  # plus the two returned parts.
  function_response_event = events[1]
  assert len(function_response_event.content.parts) == 3
  assert function_response_event.content.parts[0].function_response
  assert function_response_event.content.parts[1].inline_data
  assert function_response_event.content.parts[2].text == 'A photo of a sunset.'

  # Verify the LLM request received all parts.
  llm_request_contents = mock_model.requests[1].contents
  last_user_content = llm_request_contents[-1]
  assert last_user_content.role == 'user'
  assert len(last_user_content.parts) == 3


def test_function_returning_dict_unchanged():
  """Test that dict returns still work as before."""
  function_call = types.Part.from_function_call(
      name='get_data', args={'key': 'value'}
  )
  function_response = types.Part.from_function_response(
      name='get_data', response={'result': 'data'}
  )

  responses = [
      function_call,
      'The data is: data',
  ]
  mock_model = testing_utils.MockModel.create(responses=responses)

  def get_data(key: str) -> dict:
    """Returns data."""
    return {'result': 'data'}

  agent = Agent(name='root_agent', model=mock_model, tools=[get_data])
  runner = testing_utils.InMemoryRunner(agent)
  events = runner.run('get data')

  # The function response event should contain only the function response.
  function_response_event = events[1]
  assert len(function_response_event.content.parts) == 1
  assert function_response_event.content.parts[0].function_response


def test_function_returning_string_unchanged():
  """Test that string returns are still wrapped in result dict."""
  function_call = types.Part.from_function_call(
      name='greet', args={'name': 'Alice'}
  )

  responses = [
      function_call,
      'Hello!',
  ]
  mock_model = testing_utils.MockModel.create(responses=responses)

  def greet(name: str) -> str:
    """Greets a person."""
    return f'Hello, {name}!'

  agent = Agent(name='root_agent', model=mock_model, tools=[greet])
  runner = testing_utils.InMemoryRunner(agent)
  events = runner.run('greet alice')

  # The function response event should contain only the function response.
  function_response_event = events[1]
  assert len(function_response_event.content.parts) == 1
  assert function_response_event.content.parts[0].function_response
  assert function_response_event.content.parts[
      0
  ].function_response.response == {'result': 'Hello, Alice!'}
