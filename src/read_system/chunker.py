import re
from typing import List

def chunk_text(text: str, chunk_size: int = 10, overlap: int = 64) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    def split_sentences(p: str) -> List[str]:
        return re.split(r'(?<=[.!?])\s+', p)

    sentences: List[str] = []
    for p in paragraphs:
        sentences.extend(split_sentences(p))

    chunks = []
    cur = []
    cur_len = 0
    i = 0
    while i < len(sentences):
        s = sentences[i]
        cur.append(s)
        cur_len += len(s.split())
        i += 1
        if cur_len >= chunk_size:
            chunks.append(" ".join(cur))
            # overlap words from end
            cur = cur[-(overlap//10):]  # approximate overlap by words
            cur_len = sum(len(x.split()) for x in cur)

    if cur:
        chunks.append(" ".join(cur))

    return chunks

if __name__ == "__main__":
    longtext = """How are you today ? Can I have a cup of coffee ?
                No, you can't; I don't sell coffee.
    """

    chunks = chunk_text(longtext)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:\n{chunk}\n")