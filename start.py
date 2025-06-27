#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:author:     Vadym Stupakov <vadim.stupakov@gmail.com>
:license:    MIT
"""

import atexit
import os
import signal
import sys
from argparse import ArgumentParser
from pathlib import Path
from platform import system
from shlex import join, split
from subprocess import Popen, run
from typing import Any, Callable, TypeVar

CURR_DIR = Path(__file__).parent
PYTHON_EXE = (CURR_DIR / ".venv/bin/python3").as_posix()

T = TypeVar("T")


def kill_all():
    cmd = split(f"pkill -fe {PYTHON_EXE}")
    run(cmd)


def set_ulimit(n=4096):
    if system().lower() == "linux":
        import resource

        resource.setrlimit(resource.RLIMIT_NOFILE, (n, n))


def pipe(data: T, *functions: Callable[[T], T]) -> T:
    for f in functions:
        data = f(data)

    return data


def getenv() -> dict[str, str]:
    return dict(os.environ)


def clear_env(predicate: Callable[[str, str], bool]):
    def foo(env: dict[str, str]):
        return {k: v for k, v in env.items() if predicate(k, v)}

    return foo


def tostr(d: dict[Any, Any]) -> dict[str, str]:
    return {str(k): str(v) for k, v in d.items()}


def set_env(vars: dict[str, str]):
    def foo(env: dict[str, str]):
        env = env.copy()
        env.update(tostr(vars))
        return env

    return foo


def identity(v: T) -> T:
    return v


def split_args_on_delimetr(args: list[str], delimetr: str = "---"):
    idx = args.index(delimetr)
    return args[:idx], args[idx + 1 :]


def main():
    start_args, prog_args = split_args_on_delimetr(sys.argv, "---")

    parser = ArgumentParser()
    parser.add_argument("--debug", "-d", action="store_true")
    parser.add_argument("--profile", "-p", action="store_true")
    args = parser.parse_args(start_args[1:])

    atexit.register(kill_all)

    vars = dict(
        PYTHONPATH=CURR_DIR.as_posix(),
        PYTHONNOUSERSITE="1",
        POLARS_MAX_THREADS="10",
        TORCH_WARM_POOL="0",
    )
    debug_vars = dict(
        PYTORCH_JIT="0",
        CUDA_LAUNCH_BLOCKING="1",
    )

    env = pipe(
        getenv(),
        clear_env(lambda k, v: not k.lower().startswith("python")),
        set_env(vars),
        set_env(debug_vars) if args.debug else identity,
    )

    default_prefix = [PYTHON_EXE]
    profile_prefix = split(f"""{PYTHON_EXE}
                            -m scalene --program-path={CURR_DIR.as_posix()}
                            --no-browser --cli --cpu --json --outfile prof.json
                            --- """)

    debug_prefix = split(f"""{PYTHON_EXE}
                          -Xfrozen_modules=off
                          -m debugpy
                          --wait-for-client
                          --listen 5678
                          --configure-subProcess True""")

    if args.debug:
        prefix = debug_prefix
    elif args.profile:
        prefix = profile_prefix
    else:
        prefix = default_prefix

    cmd = prefix + prog_args
    print("cmd:", join(cmd))

    set_ulimit()
    p = Popen(cmd, env=env, cwd=CURR_DIR.as_posix())
    try:
        p.wait()
    except KeyboardInterrupt:
        p.send_signal(signal.SIGINT)
        p.wait()


main()
