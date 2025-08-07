from datetime import datetime

def reset_log(string: str = ""):
    return
    with open("test/log.txt", "w") as f:
        f.write(string)

def log_header(title: str, width: int = 60, char: str = '='):
    bar = char * width
    ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{bar}\n{ts} | {title.center(width - len(ts) - 3)}\n{bar}\n", flush=True)


def log_subheader(title: str, width: int = 50, char: str = '-'):
    bar = char * width
    print(f"{bar}\n{title.center(width)}\n{bar}", flush=True)


def log_subsubheader(title: str, width: int = 40, char: str = '~'):
    bar = char * width
    print(f"{bar}\n{title.center(width)}\n{bar}", flush=True)