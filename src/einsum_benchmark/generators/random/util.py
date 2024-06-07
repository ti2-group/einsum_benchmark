def get_symbol(i: int) -> str:
    """Get a symbol (str) corresponding to int i.

    This function runs through the usual 52 letters before using other unicode characters.
    """
    alphabetic_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    if i < 52:
        return alphabetic_symbols[i]
    return chr(i + 140)
