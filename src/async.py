import asyncio
import random

ENGINE_PATH = "../../Mzinga.LinuxX64/MzingaEngine"
DBG = True
MAX_MOVES = 10

OK = "ok\n"

async def send(writer, reader, command: str, dbg: bool = DBG) -> str:
    writer.write(f"{command}\n".encode())
    await writer.drain()
    if dbg:
        print(f"\n[->] {command}")
    if command != "exit":
        return await read_all(reader, dbg)
    else:
        return ""

async def readuntil(reader: asyncio.StreamReader, delim: str) -> str:
    output = []
    while True:
        line = await reader.readline()
        if not line:
            break
        line = line.decode().strip()
        output.append(line)
        if line.endswith(delim.strip()):
            break
    return "\n".join(output)

async def read_all(reader: asyncio.StreamReader, dbg: bool = DBG) -> str:
    output = await readuntil(reader, OK)
    for line in output.split("\n"):
        if dbg:
            print(f"[<-] {line}")
    return output

async def main():
    print("Starting interaction with MzingaEngine...\n")
    process = await asyncio.create_subprocess_exec(
        ENGINE_PATH,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    reader, writer = process.stdout, process.stdin

    # Initial read to handle any startup messages
    await read_all(reader)

    await send(writer, reader, "newgame Base+MLP")

    for _ in range(MAX_MOVES):
        moves = (await send(writer, reader, "validmoves")).split(";")
        move = random.choice(moves)
        await send(writer, reader, f"play {move}")
        await send(writer, reader, "bestmove time 00:00:02")

    await send(writer, reader, "exit")

    writer.close()
    await writer.wait_closed()
    await process.wait()

if __name__ == "__main__":
    asyncio.run(main())
