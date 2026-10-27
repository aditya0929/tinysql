"""A prefix-checking grammar for the SQL subset the model writes (WikiSQL style), over raw bytes."""

PREFIX = -1                        # parser marker: the input ended inside this construct
AGGS = (b"MAX", b"MIN", b"COUNT", b"SUM", b"AVG")
OPS = (b"=", b">", b"<")
MAX_CONDITIONS = 6


class SqlGrammar:
    def __init__(self, columns, question=None, max_value_bytes: int = 100):
        """columns: the table's column names (as the model sees them in the schema).
        question: if given, every condition value must be a (case-insensitive) substring of it.
        max_value_bytes: without a question, caps value length so a model cannot ramble inside a value forever."""
        self.cols = [('"' + c.replace('"', '""') + '"').encode("utf-8") for c in columns]
        self.q = question.lower() if question is not None else None
        # multi-byte characters a value may use (any case), for looking ahead over an unfinished character
        self.q_chars = {c for c in (question or "") + (question or "").upper() + (question or "").lower() if len(c.encode("utf-8")) > 1}
        self.max_value_bytes = max_value_bytes

    # ----------------------------------------------------------------------------- small parsers
    # Each yields end positions of a successful parse starting at i, or PREFIX if s ends inside it.
    @staticmethod
    def _lit(s, i, lit):
        rest = len(s) - i
        if rest >= len(lit):
            if s[i:i + len(lit)] == lit:
                yield i + len(lit)
        elif lit.startswith(s[i:]):
            yield PREFIX

    def _col(self, s, i):
        for c in self.cols:
            yield from self._lit(s, i, c)

    def _sel(self, s, i):
        yield from self._col(s, i)
        for agg in AGGS:
            for j in self._lit(s, i, agg + b"("):
                if j == PREFIX:
                    yield PREFIX
                    continue
                for k in self._col(s, j):
                    if k == PREFIX:
                        yield PREFIX
                        continue
                    yield from self._lit(s, k, b")")

    @staticmethod
    def _utf8_prefix(content: bytes):
        """-> (text, complete) if content is valid UTF-8, allowing only an unfinished character at the very end;
        None if it contains bytes that can never form valid UTF-8 (e.g. a stray continuation byte)."""
        try:
            return content.decode("utf-8"), True
        except UnicodeDecodeError as e:
            if e.reason == "unexpected end of data":
                return content[:e.start].decode("utf-8"), False
            return None

    def _content_ok(self, content: bytes, final: bool) -> bool:
        if final and not content:
            return False                                    # values are never empty
        decoded = self._utf8_prefix(bytes(content))
        if decoded is None or (final and not decoded[1]):
            return False                                    # garbage bytes, or a value ending mid-character
        if self.q is None:
            return len(content) <= self.max_value_bytes
        text, complete = decoded
        if complete:                                        # Unicode-aware lowercase: 'Ş' must match 'ş'
            return text.lower() in self.q
        # an unfinished character: viable only if some character of the question starts with these bytes and keeps
        # the value a substring of the question (otherwise decoding would reach a dead end)
        tail = bytes(content)[len(text.encode("utf-8")):]
        return any(c.encode("utf-8").startswith(tail) and (text + c).lower() in self.q for c in self.q_chars)

    def _value(self, s, i):
        for j in self._lit(s, i, b"'"):
            if j == PREFIX:
                yield PREFIX
                return
            k, content = j, bytearray()
            while k < len(s):
                if s[k] == 0x27:                            # a quote: closing, or the first half of an escaped ''
                    if k + 1 < len(s) and s[k + 1] == 0x27:
                        content += b"'"
                        k += 2
                    elif k + 1 == len(s):                   # ambiguous at the very end: allow both readings
                        if self._content_ok(content, final=True):
                            yield k + 1
                        if self._content_ok(content + b"'", final=False):
                            yield PREFIX
                        return
                    else:
                        if self._content_ok(content, final=True):
                            yield k + 1
                        return
                    if not self._content_ok(content, final=False):
                        return
                    continue
                content.append(s[k])
                k += 1
                if not self._content_ok(content, final=False):
                    return                                  # can no longer become a substring of the question
            yield PREFIX                                    # input ended inside the value

    def _cond(self, s, i):
        for j in self._col(s, i):
            if j == PREFIX:
                yield PREFIX
                continue
            for k in self._lit(s, j, b" "):
                if k == PREFIX:
                    yield PREFIX
                    continue
                for op in OPS:
                    for m in self._lit(s, k, op + b" "):
                        if m == PREFIX:
                            yield PREFIX
                            continue
                        yield from self._value(s, m)

    def _end(self, s, i):
        """';' that must be the very last byte."""
        for j in self._lit(s, i, b";"):
            if j == PREFIX:
                yield "prefix"
            elif j == len(s):
                yield "complete"

    def _conditions(self, s, i, depth=1):
        for j in self._cond(s, i):
            if j == PREFIX:
                yield "prefix"
                continue
            yield from self._end(s, j)
            if depth < MAX_CONDITIONS:
                for k in self._lit(s, j, b" AND "):
                    if k == PREFIX:
                        yield "prefix"
                    else:
                        yield from self._conditions(s, k, depth + 1)

    def _outcomes(self, s):
        for i in self._lit(s, 0, b"SELECT "):
            if i == PREFIX:
                yield "prefix"
                continue
            for j in self._sel(s, i):
                if j == PREFIX:
                    yield "prefix"
                    continue
                for k in self._lit(s, j, b" FROM t"):
                    if k == PREFIX:
                        yield "prefix"
                        continue
                    yield from self._end(s, k)
                    for m in self._lit(s, k, b" WHERE "):
                        if m == PREFIX:
                            yield "prefix"
                        else:
                            yield from self._conditions(s, m)

    def check(self, s: bytes) -> str:
        best = "invalid"
        for outcome in self._outcomes(s):
            if outcome == "complete":
                return "complete"
            best = "prefix"
        return best
