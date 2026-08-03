import asyncio

from qwen_reverse import Conversation

TOKEN = None

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Obtiene el clima actual de una ciudad",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


async def main():
    conv = Conversation(token=TOKEN, tools=[WEATHER_TOOL])
    async for event in conv.stream("Que tiempo hace en Buenos Aires?"):
        if event["type"] == "tool_calls":
            print(f"[tool_calls] {event['data']}")
        elif event["type"] == "content":
            print(event["data"], end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
