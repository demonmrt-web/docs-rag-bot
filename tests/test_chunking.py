from app.chunking import chunk_text


def test_empty_text():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_short_text_is_one_chunk():
    assert chunk_text("Привет, мир.", 800, 150) == ["Привет, мир."]


def test_chunks_do_not_exceed_limit_much():
    text = "\n\n".join(f"Абзац номер {i}. " + "слово " * 60 for i in range(20))
    chunks = chunk_text(text, 500, 100)
    assert len(chunks) > 3
    # допускаем перекрытие поверх лимита, но не больше размера перекрытия + разделителя
    assert all(len(c) <= 500 + 100 + 5 for c in chunks)


def test_overlap_repeats_tail_of_previous_chunk():
    text = "\n\n".join(f"Предложение {i} содержит несколько слов для проверки." for i in range(30))
    chunks = chunk_text(text, 200, 60)
    assert len(chunks) > 2
    tail_words = chunks[0].split()[-3:]
    assert all(w in chunks[1] for w in tail_words)


def test_long_sentence_without_dots_is_split_by_words():
    text = "слово " * 500
    chunks = chunk_text(text, 300, 0)
    assert len(chunks) > 5
    assert all(len(c) <= 300 for c in chunks)
    assert all(not c.startswith(" ") and not c.endswith(" ") for c in chunks)


def test_no_text_is_lost_without_overlap():
    words = [f"w{i}" for i in range(400)]
    chunks = chunk_text(" ".join(words), 120, 0)
    joined = " ".join(chunks).split()
    assert joined == words
