
# Table of Contents

1.  [Handout](#org13dd2d2)
    1.  [Problem Set 1 (unicode1)](#org0923e09)
    2.  [Problem Set 2 (unicode2)](#org41ef07f)
    3.  [Pretokenization](#org2f1b8b9)
    4.  [Special Tokens](#org76801de)
2.  [Readings](#org6b3e020)
    1.  [Karpathy Video](#org6166b6a)



<a id="org13dd2d2"></a>

# Handout


<a id="org0923e09"></a>

## Problem Set 1 (unicode1)

1.  Which unicode character does chr(0) return?
    
        print(chr(0))

2.  How does this character’s string representation (\_<sub>repr</sub>\_<sub>()</sub>) differ from its printed representation?
    
        print(chr(0).__repr__())

3.  What happens when this character occurs in text? It may be helpful to play around with the following in your Python interpreter and see if it matches your expectations:
    
        chr(0)
        print(chr(0))
        "this is a test" + chr(0) + "string"
        print("this is a test" + chr(0) + "string")
    
    How strange it seems to be some kind of empty space.


<a id="org41ef07f"></a>

## Problem Set 2 (unicode2)

1.  What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input stringsA
    
    My intuition is UTF-8 is both better for memory.
    
        a = "hello"
        print(a.encode("utf-8"))
        print(a.encode("utf-16"))
        print(a.encode("utf-32"))

2.  Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results. 
    
        def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
            print("".join([bytes([b]).decode("utf-8") for b in bytestring]))
        
        decode_utf8_bytes_to_str_wrong("hello".encode("utf-8")) 
        decode_utf8_bytes_to_str_wrong("🐂".encode("utf-8"))

Some characters like emojis require multiple bytes to represent themselves in `utf-8`

1.  Give a two byte sequence that does not decode to any Unicode character(s).
    
        a = bytes([255, 255])
        print(a)
        a.decode("utf-8")

    b'\xff\xff'
    Traceback (most recent call last):
     File "<stdin>", line 3, in <module>
    UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte

Unicode uses the last bit as a signal about how many bytes there are left.


<a id="org2f1b8b9"></a>

## Pretokenization

1.  Added a `Pool.starmap`, which allows me to pass in an iterator for multiple arguments and spawn processes based on that.
2.  Had to write something to merge all the count<sub>maps</sub>&#x2026; right now just iterating through them but there might be a better way?
3.  Refactored function into something that can work given just the start and end, had to do some global scoping


<a id="org76801de"></a>

## Special Tokens

1.  Need to find a regex that splits the text into distinct pretokens that should be processed. In other words, upon seeing a `"<|endoftext|>"`, it should be preserved as its own token in the final vocabulary, as well as be used to split specifically on.
    
        import regex as re
        special_tokens = ["<|endoftext|>"]
        text = "[Doc 1] Some text here<|endoftext|>[Doc 2] More text End"
        escaped_special_tokens: list[str] = [re.escape(token) for token in special_tokens]
        special_token_pattern = "|".join(escaped_special_tokens)
        split_chunk_list: list[str] = re.split(f"({special_token_pattern})", text)
        print(split_chunk_list)

2.  Needed to write some logic to &rsquo;skip&rsquo; special tokens, also reserve an index for them. So I added some extra code to to the training step. `special_map` will map special<sub>tokens</sub> to their corresponding ids, so that it&rsquo;s useful at the pretoken sequence building step to skip them.
    
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

3.  Had a really gnarly indentation error in my code. I wrote the counting step such that the max value would be fetched every iteration rather than at the end. Calling &rsquo;max&rsquo; so many times meant that my tests took about 35 minutes to complete. I thought that was normal until I read the docstring for the test saying it should take less than 1.5 seconds. Anyways it&rsquo;s fixed now and my speed is &asymp; 1.5 seconds now.
4.  


<a id="org6b3e020"></a>

# Readings


<a id="org6166b6a"></a>

## Karpathy Video

[Let&rsquo;s build the GPT Tokenizer - YouTube](https://youtu.be/zduSFxRajkE)
Byte Pair Encoding is great to compress text in a way that&rsquo;s manageable for the attention block. The core idea is straightforward, merge character pairs that commonly occur with each other and do this recursively. The merges are then tracked using a lookup table.

    aaabdaaabac -> ZabdZabac
    Z = aa
    
    ZabdZabac -> ZYdZYac
    Y = ab
    Z = aa
    
    ZYdZYac -> XdXac
    X = ZY
    Y = ab
    Z = aa

Decompressing this involves finding a character in the compressed string, replacing it with the value in the lookup table and doing so until you end up with no other lookups possible.

Something unintuitive about Unicode is that it sometimes takes **multiple** bytes to represent a single character. In hindsight, this makes sense since there are way too many languages and characters to represent and going past 255 isn&rsquo;t really possible with bytes.

**Fun tangent:** I wondered how multiple bytes can be expressed using unicode because in my head, I thought that all 255 numbers are used to represent some character and then they keep going. Singular bytes stop at 127.
Turns out that the leading bits are a &ldquo;signal&rdquo; that this is a &ldquo;multi-byte&rdquo; Unicode code point and should be read as such.

