import asyncio

from qwen_reverse import Video

TOKEN = None


async def main():
    vid = Video(token=TOKEN)
    urls = await vid.generate(
        "Un gato astronauta flotando en el espacio, cámara lenta",
        aspect_ratio="16:9",
    )
    for url in urls:
        print(url)


if __name__ == "__main__":
    asyncio.run(main())
