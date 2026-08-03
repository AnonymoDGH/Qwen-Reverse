import asyncio

from qwen_reverse import Image

TOKEN = None


async def main():
    img = Image(token=TOKEN)
    urls = await img.generate(
        "Un dragón cyberpunk volando sobre una ciudad neón, estilo anime",
        aspect_ratio="16:9",
    )
    for url in urls:
        print(url)


if __name__ == "__main__":
    asyncio.run(main())
