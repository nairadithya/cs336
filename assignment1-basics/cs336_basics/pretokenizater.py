import os
from abc import ABC
from dataclasses import dataclass
from typing import BinaryIO
import regex as re  # ty:ignore[unresolved-import]
from multiprocessing import Pool

def merge(
    indices: list[int], pair: tuple[int, int], new_index: int
) -> list[int]:
    """Return `indices`, but with all instances of `pair` replaced with `new_index`."""
    new_indices = []
    i = 0
    while i < len(indices):
        if (
            i + 1 < len(indices)
            and indices[i] == pair[0]
            and indices[i + 1] == pair[1]
        ):
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def chunk_processer(start: int, end: int, input_path: str, special_tokens: list[str]):
    with open(input_path, "rb") as f:
        f.seek(start)
        file_chunk = f.read(end - start).decode("utf-8", errors="ignore")
        escaped_special_tokens: list[str] = [re.escape(token) for token in special_tokens]
        special_token_pattern = "|".join(escaped_special_tokens)
        split_chunk_list: list[str] = re.split(f"({special_token_pattern})", file_chunk)
        # GPT-2 regex
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        count_map: dict[str, int] = {}
        for chunk in split_chunk_list:
            pretokens = re.finditer(PAT, chunk)
            for item in pretokens:
                count_map[item.group(0)] = count_map.get(item.group(0), 0) + 1
    return count_map

def run_train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
        num_processes = 32
        final_count_map: dict[str, int]= {}
        with open(input_path, mode="rb") as f:
            boundaries: list[int] = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        with Pool(num_processes) as pool:
            count_maps: list[dict[str,int]] = pool.starmap(chunk_processer,[(start, end, input_path, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])])
            for map in count_maps:
                for pretoken in map:
                    final_count_map[pretoken] = final_count_map.get(pretoken, 0) + map[pretoken]

        def train_bpe_pretokenization(count_map: dict[str, int], num_merges: int):
            merges: dict[tuple[int, int], int] = {}
            vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}

            special_map: dict[str, int] = {}
            for i, special_token in enumerate(special_tokens):
                byte_special = special_token.encode('utf-8')
                vocab[256+i] = byte_special
                special_map[special_token] = 256 + i
            pretoken_sequences: dict[str, list[int]] = {}
            for pretoken in count_map:
                if pretoken in special_tokens:
                    reserved_index = special_map[pretoken]
                    pretoken_sequences[pretoken] = [reserved_index]
                else:
                    pretoken_sequences[pretoken] = list(pretoken.encode("utf-8"))
            for i in range(num_merges):
                counts: dict[tuple[int, int], int] = {}
                for pretoken, indices in pretoken_sequences.items():
                    for index1, index2 in zip(indices, indices[1:]):
                        counts[(index1, index2)] = (counts.get((index1, index2), 0) + count_map[pretoken])
                pair = max(counts, key=counts.get)  # ty:ignore[no-matching-overload]
                index1, index2 = pair
                new_index = 256 + len(special_tokens) + i
                merges[pair] = new_index
                vocab[new_index] = (vocab[index1] + vocab[index2])  

            for pretoken in pretoken_sequences:
                pretoken_sequences[pretoken] = merge(pretoken_sequences[pretoken], pair, new_index)

            return vocab, merges
        return train_bpe_pretokenization(final_count_map, vocab_size - 256 - len(special_tokens))
