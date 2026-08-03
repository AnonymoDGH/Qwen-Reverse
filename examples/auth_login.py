import asyncio

from qwen_reverse import start_device_login, complete_device_login


async def main():
    client, data = await start_device_login()
    print(f"Visita: {data['verification_uri']}")
    print(f"Codigo: {data['user_code']}")
    token = await complete_device_login(client, data)
    print(f"Token guardado: {token[:16]}...")


if __name__ == "__main__":
    asyncio.run(main())
