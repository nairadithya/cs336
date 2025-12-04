import os
from abc import ABC
from dataclasses import dataclass
from typing import BinaryIO

import regex as re  # ty:ignore[unresolved-import]

# Taken from the handout
class Tokenizer(ABC):
    """Abstract interface for a tokenizer."""

    def encode(self, string: str) -> list[int]:
        raise NotImplementedError

    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class BPETokenizerParams:
    """All you need to specify a BPETokenizer."""

    vocab: dict[int, bytes]  # index -> bytes
    merges: dict[tuple[int, int], int]  # index1,index2 -> new_index


class BPETokenizer(Tokenizer):
    """BPE tokenizer given a set of merges and a vocabulary."""

    def __init__(self, params: BPETokenizerParams):
        self.params = params

    def encode(self, string: str) -> list[int]:
        indices = list(map(int, string.encode("utf-8")))
        for pair, new_index in self.params.merges.items():
            indices = merge(indices, pair, new_index)
        return indices

    def decode(self, indices: list[int]) -> str:
        bytes_list = list(map(self.params.vocab.get, indices))
        string = b"".join(bytes_list).decode("utf-8")  # @inspect string
        return string


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


def train_bpe_pretokenization(
    count_map: dict[str, int], num_merges: int
) -> BPETokenizerParams:
    pretoken_sequences: dict[str, list[int]] = {}
    for pretoken in count_map:
        pretoken_sequences[pretoken] = list(pretoken.encode("utf-8"))

    merges: dict[tuple[int, int], int] = {}
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}
    for i in range(num_merges):
        counts: dict[tuple[int, int], int] = {}

        for pretoken, indices in pretoken_sequences.items():
            for index1, index2 in zip(indices, indices[1:]):
                counts[(index1, index2)] = (
                    counts.get((index1, index2), 0) + count_map[pretoken]
                )
                pair = max(counts, key=counts.get)  # ty:ignore[no-matching-overload]
                index1, index2 = pair
                new_index = 256 + i
                merges[pair] = new_index
                vocab[new_index] = (
                    vocab[index1] + vocab[index2]
                )  # @inspect vocab

        for pretoken in pretoken_sequences:
            pretoken_sequences[pretoken] = merge(
                pretoken_sequences[pretoken], pair, new_index
            )

    return BPETokenizerParams(vocab=vocab, merges=merges)


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

from multiprocessing import Pool

input_path = "notes.org"

def process_chunk(start: int, end: int) -> dict[str, int]:
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        count_map: dict[str, int] = {}
        # Pre-tokenization
        pretokens: list[str] = re.finditer(PAT, chunk)
        for item in pretokens:
            count_map[item[0]] = count_map.get(item[0], 0) + 1
        return count_map

def run_train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
        num_processes = 8
        with open(input_path, mode="rb") as f:
            boundaries: list[int] = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        with Pool(num_processes) as pool:
            count_maps: list[dict[str,int]] = pool.starmap(process_chunk, [i for i in zip(boundaries[:-1], boundaries[1:])])
            final_count_map: dict[str, int]= {}
            for map in count_maps:
                for pretoken in map:
                    final_count_map[pretoken] = final_count_map.get(pretoken, 0) + map[pretoken]
        params = train_bpe_pretokenization(final_count_map, 300)


string = "notes.org"
vocab_size = 200
special_tokens = ["<|end_of_text|>"]


params = run_train_bpe(string, vocab_size, special_tokens)
print(params)
