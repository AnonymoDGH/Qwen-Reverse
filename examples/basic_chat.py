import asyncio

from qwen_reverse import Generative

TOKEN = None  # o tu token de chat.qwen.ai


async def main():
    gen = Generative(token=TOKEN)

    print("=== one-shot ===")
    text = await gen.generate("Explica en 2 lineas que es un kernel")
    print(text)

    print("\n=== streaming ===")
    async for event in gen.stream("Cuenta del 1 al 5", reasoning_effort="none"):
        if event["type"] == "reasoning":
            print(f"[reasoning] {event['data']}", end="")
        elif event["type"] == "content":
            print(event["data"], end="", flush=True)
        elif event["type"] == "usage":
            print(f"\n[usage] {event['data']}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
