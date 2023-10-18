import asyncio
import websockets


async def websocket_client():
    uri = "ws://127.0.0.1:22335"
    async with websockets.connect(uri) as websocket:
        print("Connected to server.")
        async for message in websocket:
            print(f"Received message: {message}")


asyncio.run(websocket_client())
