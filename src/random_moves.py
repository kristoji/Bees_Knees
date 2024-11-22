from pwn import *

ENGINE_PATH = "../../Mzinga.LinuxX64/MzingaEngine"
DBG = True

def send(p, command: str, dbg=DBG) -> str:
    """
    Sends a command to the MzingaEngine and returns its output.
    """

    p.sendline(command.encode())
    if dbg:
        print(f"[->] {command}")
    if command != "exit":
        return read_all(p, dbg)
    else:
        return b""


def read_all(p, dbg=DBG) -> str:
    stdout = b""
    line = b""
    while line != b"ok\n":
        line = p.readline()
        if dbg:
            print(f"[<-] {line.decode().strip()}")
        stdout += line
    return stdout

if __name__ == "__main__":
    print("Starting interaction with MzingaEngine...\n")
    p = process(ENGINE_PATH)
    read_all(p, DBG)

    send(p, "info")
    send(p, "newgame")
    send(p, "validmoves")
    send(p, "exit")
    p.close()


