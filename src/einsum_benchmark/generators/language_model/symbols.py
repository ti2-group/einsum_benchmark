from typing import Iterable


EXTRA_SYMBOLS = "".join([chr(pos) for pos in range(193, 10_000)])
DIGITS = "0123456789"
LOWERCASE_LETTERS = "abcdefghijklmnopqrstuvwxyz"
UPPERCASE_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SYMBOLS = LOWERCASE_LETTERS + UPPERCASE_LETTERS + DIGITS + EXTRA_SYMBOLS


def remaining_symbols(used_symbols: str | list[str], original_symbols=SYMBOLS) -> str:
    if isinstance(used_symbols, Iterable):
        used_symbols = "".join(used_symbols)
    return "".join([symbol for symbol in original_symbols if symbol not in used_symbols])


class SymbolGenerator:
    def __init__(self, all_symbols: str = SYMBOLS) -> None:
        self.all_symbols = all_symbols
        self.n_used_symbols = 0

    def generate(self, n_symbols: int = 1) -> str:
        if self.n_used_symbols + n_symbols > len(self.all_symbols):
            raise IndexError("not enough symbols.")
        generated_symbols = self.all_symbols[self.n_used_symbols: self.n_used_symbols + n_symbols]
        self.n_used_symbols += n_symbols
        return generated_symbols

    def used_symbols(self) -> str:
        return self.all_symbols[:self.n_used_symbols]
