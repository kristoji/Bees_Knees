from datetime import datetime
import functools

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


total_calls = 0
def countit(fn):
  @functools.wraps(fn)
  def wrapper(*args, **kwargs):
    global total_calls
    try:
        return fn(*args, **kwargs)
    finally:
        total_calls += 1
  return wrapper

def print_counter():
    global total_calls
    print(f"[counter] total calls: {total_calls}")
