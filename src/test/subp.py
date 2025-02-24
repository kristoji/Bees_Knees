import subprocess

# TODO: add check for win/loss/draw

DBG = False
MZINGA_PATH = "../../../MzingaEngine"
OTHER_PATH = "../../dist/BeesKneesEngineAI"

MAX_TURNS = 5
DEPTH = 1
TIME_TOTAL_SEC = 2


TIME_H = TIME_TOTAL_SEC // 3600
TIME_M = TIME_TOTAL_SEC // 60
TIME_S = TIME_TOTAL_SEC % 60
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

def start_process(path) -> subprocess.Popen:
    return subprocess.Popen(
        [path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        universal_newlines=True,
    )

def end_process(p: subprocess.Popen) -> None:
    p.stdin.close()
    p.stdout.close()
    p.stderr.close()
    p.kill()

def play_step(p1: subprocess.Popen, p2: subprocess.Popen, time: bool = False) -> str:
    if time:
        move = send(p1, f"bestmove time {TIME_H:02}:{TIME_M:02}:{TIME_S:02}")
    else:
        move = send(p1, f"bestmove depth {DEPTH}")

    move = move.strip().split("\n")[0]
    name = p1.args[0].split("/")[-1][:12]
    print(f"[{name}] \tplays: {move}")
    
    send(p1, f"play {move}")
    return send(p2, f"play {move}")

def check_end_game(out: str) -> None:
    return "InProgress" != out.split(";")[1]


if __name__ == "__main__":
    
    print("Starting interaction with MzingaEngine...\n")
    mzinga: subprocess.Popen = start_process(MZINGA_PATH)
    read_all(mzinga)

    print(f"Starting interaction with {OTHER_PATH.split("/")[-1][:12]}...\n")
    p = start_process(OTHER_PATH)
    read_all(p)

    send(mzinga, "newgame Base+MLP")
    send(p, "newgame Base+MLP")

    for i in range(MAX_TURNS):
        out = play_step(mzinga, p)
        if check_end_game(out):
            print(out)
            break

        out = play_step(p, mzinga)
        if check_end_game(out):
            print(out)
            break
        
        print("Board: " + ";".join(out.split("\n")[0].split(";")[3:]) + "\n")

    send(mzinga, "exit")
    send(p, "exit")

    
    end_process(mzinga)
    end_process(p)
    print("Done.")
    
    
