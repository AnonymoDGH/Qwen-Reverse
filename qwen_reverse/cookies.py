import random
import time
from typing import Any, Callable, Dict, List, Optional

from .fingerprint import generate_fingerprint

CUSTOM_BASE64_CHARS = "DGi0YA7BemWnQjCl4_bR3f8SKIF9tUz/xhr2oEOgPpac=61ZqwTudLkM5vHyNXsVJ"

HASH_FIELDS: Dict[int, str] = {
    16: "split",
    17: "full",
    18: "full",
    31: "full",
    34: "full",
    36: "full",
}

def lzw_compress(data: Optional[str], bits: int, char_func: Callable[[int], str]) -> str:
    if data is None:
        return ""
    dictionary: Dict[str, int] = {}
    dict_to_create: Dict[str, bool] = {}
    c = ""
    wc = ""
    w = ""
    enlarge_in = 2
    dict_size = 3
    num_bits = 2
    result: List[str] = []
    value = 0
    position = 0
    for i in range(len(data)):
        c = data[i]
        if c not in dictionary:
            dictionary[c] = dict_size
            dict_size += 1
            dict_to_create[c] = True
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            if w in dict_to_create:
                if ord(w[0]) < 256:
                    for _ in range(num_bits):
                        value = (value << 1)
                        if position == bits - 1:
                            position = 0
                            result.append(char_func(value))
                            value = 0
                        else:
                            position += 1
                    char_code = ord(w[0])
                    for _ in range(8):
                        value = (value << 1) | (char_code & 1)
                        if position == bits - 1:
                            position = 0
                            result.append(char_func(value))
                            value = 0
                        else:
                            position += 1
                        char_code >>= 1
                else:
                    char_code = 1
                    for _ in range(num_bits):
                        value = (value << 1) | char_code
                        if position == bits - 1:
                            position = 0
                            result.append(char_func(value))
                            value = 0
                        else:
                            position += 1
                        char_code = 0
                    char_code = ord(w[0])
                    for _ in range(16):
                        value = (value << 1) | (char_code & 1)
                        if position == bits - 1:
                            position = 0
                            result.append(char_func(value))
                            value = 0
                        else:
                            position += 1
                        char_code >>= 1
                enlarge_in -= 1
                if enlarge_in == 0:
                    enlarge_in = 2 ** num_bits
                    num_bits += 1
                del dict_to_create[w]
            else:
                char_code = dictionary[w]
                for _ in range(num_bits):
                    value = (value << 1) | (char_code & 1)
                    if position == bits - 1:
                        position = 0
                        result.append(char_func(value))
                        value = 0
                    else:
                        position += 1
                    char_code >>= 1
            enlarge_in -= 1
            if enlarge_in == 0:
                enlarge_in = 2 ** num_bits
                num_bits += 1
            dictionary[wc] = dict_size
            dict_size += 1
            w = c
    if w != "":
        if w in dict_to_create:
            if ord(w[0]) < 256:
                for _ in range(num_bits):
                    value = (value << 1)
                    if position == bits - 1:
                        position = 0
                        result.append(char_func(value))
                        value = 0
                    else:
                        position += 1
                char_code = ord(w[0])
                for _ in range(8):
                    value = (value << 1) | (char_code & 1)
                    if position == bits - 1:
                        position = 0
                        result.append(char_func(value))
                        value = 0
                    else:
                        position += 1
                    char_code >>= 1
            else:
                char_code = 1
                for _ in range(num_bits):
                    value = (value << 1) | char_code
                    if position == bits - 1:
                        position = 0
                        result.append(char_func(value))
                        value = 0
                    else:
                        position += 1
                    char_code = 0
                char_code = ord(w[0])
                for _ in range(16):
                    value = (value << 1) | (char_code & 1)
                    if position == bits - 1:
                        position = 0
                        result.append(char_func(value))
                        value = 0
                    else:
                        position += 1
                    char_code >>= 1
            enlarge_in -= 1
            if enlarge_in == 0:
                enlarge_in = 2 ** num_bits
                num_bits += 1
            del dict_to_create[w]
        else:
            char_code = dictionary[w]
            for _ in range(num_bits):
                value = (value << 1) | (char_code & 1)
                if position == bits - 1:
                    position = 0
                    result.append(char_func(value))
                    value = 0
                else:
                    position += 1
                char_code >>= 1
        enlarge_in -= 1
        if enlarge_in == 0:
            enlarge_in = 2 ** num_bits
            num_bits += 1
    char_code = 2
    for _ in range(num_bits):
        value = (value << 1) | (char_code & 1)
        if position == bits - 1:
            position = 0
            result.append(char_func(value))
            value = 0
        else:
            position += 1
        char_code >>= 1
    while True:
        value = (value << 1)
        if position == bits - 1:
            result.append(char_func(value))
            break
        position += 1
    return "".join(result)

def custom_encode(data: Optional[str], url_safe: bool) -> str:
    if data is None:
        return ""
    base64_chars = CUSTOM_BASE64_CHARS
    compressed = lzw_compress(data, 6, lambda index: base64_chars[index])
    if not url_safe:
        mod = len(compressed) % 4
        if mod == 1:
            return compressed + "==="
        if mod == 2:
            return compressed + "=="
        if mod == 3:
            return compressed + "="
        return compressed
    return compressed

def random_hash() -> int:
    return random.randint(0, 0xFFFFFFFF)

def generate_cookies(
    real_data: Optional[str] = None,
    fingerprint_options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    if fingerprint_options is None:
        fingerprint_options = {}
    fingerprint = real_data or generate_fingerprint(fingerprint_options)
    fields = fingerprint.split("^")
    processed: List[Any] = list(fields)
    current_timestamp = int(time.time() * 1000)
    for idx, typ in HASH_FIELDS.items():
        if idx >= len(processed):
            continue
        if typ == "split":
            val = str(processed[idx])
            parts = val.split("|")
            if len(parts) == 2:
                processed[idx] = f"{parts[0]}|{random_hash()}"
        elif typ == "full":
            if idx == 36:
                processed[idx] = random.randint(10, 100)
            else:
                processed[idx] = random_hash()
    if 33 < len(processed):
        processed[33] = current_timestamp
    ssxmod_itna_data = "^".join(map(str, processed))
    ssxmod_itna = "1-" + custom_encode(ssxmod_itna_data, True)
    ssxmod_itna2_data = "^".join(map(str, [
        processed[0], processed[1], processed[23],
        0, "", 0, "", "", 0, 0, 0,
        processed[32], processed[33], 0, 0, 0, 0, 0
    ]))
    ssxmod_itna2 = "1-" + custom_encode(ssxmod_itna2_data, True)
    return {
        "ssxmod_itna": ssxmod_itna,
        "ssxmod_itna2": ssxmod_itna2,
        "timestamp": int(processed[33]),
        "rawData": ssxmod_itna_data,
        "rawData2": ssxmod_itna2_data,
    }
