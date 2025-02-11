import subprocess
import random

ENGINE_PATH = "../../Mzinga.LinuxX64/MzingaEngine"
DBG = True
MAX_MOVES = 10

OK = "ok\n"

def send(p: subprocess.Popen, command: str, dbg: bool=DBG) -> str:
    p.stdin.write(command + "\n")
    p.stdin.flush()
    if dbg:
        print(f"\n[->] {command}")
    if command != "exit":
        return read_all(p, dbg)
    else:
        return b""

def readuntil(p: subprocess.Popen, delim: str) -> str:
    output = []
    while True:
        line = p.stdout.readline()
        if not line:
            break
        output.append(line.strip())
        if line.endswith(delim):
            break
    return "\n".join(output)

def read_all(p: subprocess.Popen, dbg: bool=DBG) -> str:
    output = readuntil(p, OK)
    for line in output.split("\n"):
        if dbg:
            print(f"[<-] {line}")
    return output

if __name__ == "__main__":
    print("Starting interaction with MzingaEngine...\n")
    with subprocess.Popen(
        [ENGINE_PATH],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        universal_newlines=True,
    ) as p:
        read_all(p)

        # send(p, "info")
        send(p, "newgame Base+MLP")

        for i in range(MAX_MOVES):
            moves = send(p, "validmoves").split(";")
            move = random.choice(moves)
            send(p, f"play {move}")
            send(p, "bestmove time 00:00:02")

        send(p, "exit")
        
        p.stdin.close()
        p.stdout.close()
        p.stderr.close()
        p.kill()
        
