# coding: utf-8
"""Streaming readers for large JSON artifacts."""

import json


def iter_json_array(path, limit=0, chunk_size=1024 * 1024):
    """Yield objects from a large JSON array without loading the whole file."""
    decoder = json.JSONDecoder()
    buffer = ""
    yielded = 0
    array_started = False
    finished = False

    with open(path, "r", encoding="utf-8") as f:
        while not finished:
            chunk = f.read(chunk_size)
            if chunk:
                buffer += chunk
            elif not buffer.strip():
                break

            while True:
                buffer = buffer.lstrip()
                if not array_started:
                    if not buffer:
                        break
                    if buffer[0] != "[":
                        raise ValueError(f"特征文件不是 JSON 数组: {path}")
                    buffer = buffer[1:]
                    array_started = True
                    continue

                buffer = buffer.lstrip()
                if not buffer:
                    break
                if buffer[0] == "]":
                    finished = True
                    buffer = buffer[1:]
                    break
                if buffer[0] == ",":
                    buffer = buffer[1:]
                    continue

                try:
                    item, idx = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    if not chunk:
                        raise
                    break

                yield item
                yielded += 1
                if limit > 0 and yielded >= limit:
                    return
                buffer = buffer[idx:]
