import os

config_list = [
    {
        "model": "gpt-4o"
    }
]

# set autogen user agent and assistant agent with function calling
llm_config = {
    "timeout": 600,
    "seed": 42,
    "config_list": config_list,
    "temperature": 0,
}