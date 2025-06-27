#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:author:     Vadym Stupakov <vadim.stupakov@gmail.com>
:license:    MIT
"""

import socket

import psutil


def is_port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


def find_process_using_port(host: str, port: int) -> psutil.Process:
    connections = psutil.net_connections()
    for conn in connections:
        if conn.laddr.ip == host and conn.laddr.port == port:
            return psutil.Process(conn.pid)
    raise Exception(f"No process found using port {port} on {host}")


def kill_process(process: psutil.Process) -> None:
    process.terminate()
    process.wait(timeout=5)


def free_port(host: str, port: int) -> None:
    process = find_process_using_port(host, port)
    kill_process(process)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Check and kill process using specified port and host.")
    parser.add_argument("host", type=str, help="Host address to check (e.g., 127.0.0.1).")
    parser.add_argument("port", type=int, help="Port number to free.")
    args = parser.parse_args()

    if is_port_in_use(args.host, args.port):
        try:
            free_port(args.host, args.port)
            print(f"Successfully freed port {args.port} on host {args.host}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Port {args.port} is not in use on host {args.host}")


if __name__ == "__main__":
    main()
