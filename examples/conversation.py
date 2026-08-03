import asyncio

from qwen_reverse import Conversation

TOKEN = None


async def main():
    conv = Conversation(token=TOKEN)

    r1 = await conv.send("Mi nombre es Popbob y trabajo con kernels en C.")
    print(f"turno 1: {r1}\n")

    r2 = await conv.send("Como me llamo?")
    print(f"turno 2: {r2}\n")

    print(f"conversation_id: {conv.conversation_id}")
    print(f"parent_id: {conv.parent_id}")


if __name__ == "__main__":
    asyncio.run(main())
