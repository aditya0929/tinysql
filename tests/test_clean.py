from data.clean import clean_se, clean_sql, clean_web, format_gretel, normalize

GOOD_TEXT = "The water cycle describes how water moves around the earth. " * 10


def test_normalize():
    assert normalize("a\r\nb\rc\x00d\x07 ") == "a\nb\ncd"
    assert normalize("é") == "é"                      # NFC composes
    assert normalize("x²") == "x²"                      # NFC (unlike NFKC) leaves superscripts alone


def test_web_keeps_good_text_and_drops_bad():
    assert clean_web(GOOD_TEXT)[0] is not None
    assert clean_web("short")[1] == "too_short"
    assert clean_web(("ab�" * 200))[1] == "replacement_chars"
    assert clean_web("1234 5678 " * 100)[1] == "low_alpha_fraction"
    boiler = "\n".join(["Click here to subscribe to our newsletter"] * 20 + [f"unique line number {i} here" for i in range(5)])
    assert clean_web(boiler * 2 + " " * 300)[1] == "repeated_lines"


def test_sql_keeps_real_queries():
    q = "SELECT name, COUNT(*) FROM employees WHERE salary > 50000 GROUP BY department;\n" * 3
    assert clean_sql(q) == (q.strip(), None)


def test_sql_drops_non_sql_and_long_lines():
    assert clean_sql("just some prose that has no database keywords inside it at all, sorry")[1] == "no_sql_keyword"
    assert clean_sql("SELECT " + "x," * 600 + " 1 FROM t;")[1] == "long_lines"
    assert clean_sql("x")[1] == "too_short"


def test_sql_truncates_insert_dumps_but_keeps_the_schema():
    schema = "CREATE TABLE users (id INT, name TEXT);\n"
    dump = schema + "\n".join(f"INSERT INTO users VALUES ({i}, 'name{i}');" for i in range(2000))
    text, why = clean_sql(dump)
    assert why == "truncated_dump" and text.startswith("CREATE TABLE users") and len(text) <= 8000
    assert text.endswith(";")                                     # cut at a line boundary, not mid-statement


def test_stackexchange_filters():
    assert clean_se("x" * 500, 3)[0] is not None
    assert clean_se("x" * 500, -2)[1] == "negative_score"
    assert clean_se("tiny", 5)[1] == "too_short"


def test_gretel_prompt_format_keeps_only_create_table():
    ctx = "CREATE TABLE t (a INT, b TEXT); INSERT INTO t VALUES (1, 'x');"
    text, why = format_gretel(ctx, "How many rows?", "SELECT COUNT(*) FROM t;")
    assert why is None
    assert text == ("<schema>CREATE TABLE t (a INT, b TEXT);</schema>\n<question>How many rows?</question>\n"
                    "<sql>SELECT COUNT(*) FROM t;</sql>")
    assert format_gretel("INSERT INTO t VALUES (1);", "q", "SELECT 1")[1] == "missing_field"
