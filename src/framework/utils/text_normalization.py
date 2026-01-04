import argparse
import logging
import re
import unicodedata
from pathlib import Path
from typing import List, Optional, Union

from tqdm.auto import tqdm

# non-ASCII letters that are not separated by "NFKD" normalization
ADDITIONAL_DIACRITICS = {
    "œ": "oe",
    "Œ": "OE",
    "ø": "o",
    "Ø": "O",
    "æ": "ae",
    "Æ": "AE",
    "ß": "ss",
    "ẞ": "SS",
    "đ": "d",
    "Đ": "D",
    "ð": "d",
    "Ð": "D",
    "þ": "th",
    "Þ": "th",
    "ł": "l",
    "Ł": "L",
}

ER_WHITELIST = (
    "(儿女|儿子|儿孙|女儿|儿媳|妻儿|"
    "胎儿|婴儿|新生儿|婴幼儿|幼儿|少儿|小儿|儿歌|儿童|儿科|托儿所|孤儿|"
    "儿戏|儿化|台儿庄|鹿儿岛|正儿八经|吊儿郎当|生儿育女|托儿带女|养儿防老|痴儿呆女|"
    "佳儿佳妇|儿怜兽扰|儿无常父|儿不嫌母丑|儿行千里母担忧|儿大不由爷|苏乞儿)"
)
ER_WHITELIST_PATTERN = re.compile(ER_WHITELIST)

FILLER_CHARS = ["呃", "啊", "嗯"]


def remove_symbols_and_diacritics(s: str, keep=""):
    """
    Replace any other markers, symbols, and punctuations with a space,
    and drop any diacritics (category 'Mn' and some manual mappings)
    """
    # First, replace # followed by digits
    s = re.sub(r"#[\d]+", " ", s)
    return "".join(
        (
            c
            if c in keep
            else (
                ADDITIONAL_DIACRITICS[c]
                if c in ADDITIONAL_DIACRITICS
                else (
                    ""
                    if unicodedata.category(c) == "Mn"
                    else " " if unicodedata.category(c)[0] in "MSP" else c
                )
            )
        )
        for c in unicodedata.normalize("NFKD", s)
    )


def remove_symbols_only(s: str):
    """
    Replace any other markers, symbols, punctuations with a space, keeping diacritics
    """

    s = re.sub(r"#[\d]+", " ", s)
    s = "".join(
        " " if unicodedata.category(c)[0] in "MSP" and c != "'" else c
        for c in unicodedata.normalize("NFKC", s)
    )
    # carefully handle <'> for en-euro languages
    s = re.sub(r"(?<![\w])'", " ", s)
    s = re.sub(r"'(?![\w])", " ", s)
    return s


def japanese_normalize(text, mode="hira", split=True):
    import pykakasi

    """ Normalize japanese text by PyKakasi library.
        Mode: 'hira' / 'kana' / 'hepburn'
    """
    text = text.replace(" ", "")
    kk = pykakasi.kakasi()
    target = "".join(token["hira"] for token in kk.convert(text))
    if split:
        target = [i for i in target]
    return target


def to_simple(text: Union[str, List[str]]):
    """Convert traditional Chinese to simplified Chinese.
    Args:
        text: It can be a string or a list of strings.
    Returns:
        Return a string or a list of strings converted to simplified Chinese.
    """
    from zhconv import convert

    if isinstance(text, str):
        text = convert(text, "zh-cn")
        return text
    elif isinstance(text, list):
        result_text = []
        for t in text:
            t = convert(t, "zh-cn")
            result_text.append(t)
        return result_text
    else:
        raise Exception(f"Not support type{type(text)}")


def merge_single_characters(text):
    """
    Merge consecutive single English characters into a single word.
    """
    # Regular expression to match sequences of single letters
    merged_text = re.sub(
        r"\b([A-Za-z])(?:\s+([A-Za-z]))+\b", lambda m: "".join(m.group(0).split()), text
    )
    return merged_text


def remove_erhua_zh(text):
    """
    去除儿化音词中的儿:
    他女儿在那边儿 -> 他女儿在那边
    """

    new_str = ""
    while re.search("儿", text):
        a = re.search("儿", text).span()
        remove_er_flag = 0

        if ER_WHITELIST_PATTERN.search(text):
            b = ER_WHITELIST_PATTERN.search(text).span()
            if b[0] <= a[0]:
                remove_er_flag = 1

        if remove_er_flag == 0:
            new_str = new_str + text[0 : a[0]]
            text = text[a[1] :]
        else:
            new_str = new_str + text[0 : b[1]]
            text = text[b[1] :]

    text = new_str + text
    return text


def text_normalization(
    s: str,
    case: str = None,
    remove_symbols: bool = False,
    remove_diacritics: bool = False,
    space_between_cjk: bool = False,
    simplified_chinese: bool = False,
    simplified_japanese: bool = False,
    merge_single_char: bool = False,
    remove_erhua: bool = False,
    remove_fillers: bool = False,
    remove_in_brackets: bool = False,
    remove_in_parenthesis: bool = False,
    special_tokens_to_keep: List[str] = [],
):
    # Protect special tokens (e.g. "<not end>") from all normalization steps
    # by temporarily replacing them with placeholders, then restoring them.
    # Use private-use Unicode code points as placeholders. They:
    # - are very unlikely to appear in real text;
    # - are not affected by case conversion or symbol-removal (category "Co").
    placeholders = {}
    if special_tokens_to_keep:
        base_cp = 0xE000  # start of BMP private-use area
        for i, tok in enumerate(special_tokens_to_keep):
            ph = chr(base_cp + i)
            placeholders[ph] = tok
            s = s.replace(tok, ph)

    # Optionally remove content in <> or [] brackets.
    if remove_in_brackets:
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)
    if remove_in_parenthesis:
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
    if remove_symbols:
        s = remove_symbols_only(s)

    if remove_diacritics:
        s = remove_symbols_and_diacritics(s)

    if case == "lower":
        s = s.lower()
    elif case == "upper":
        s = s.upper()

    pattern = re.compile(
        r"([\u1100-\u11ff\u2e80-\ua4cf\ua840-\uD7AF\uF900-\uFAFF\uFE30-\uFE4F\uFF65-\uFFDC\U00020000-\U0002FFFF\u3000-\u303F\uff01-\uff60])"
    )
    chars = pattern.split(s)
    chars = [ch for ch in chars if ch.strip()]
    if not space_between_cjk:
        # Join with space only if the segment is not CJK
        s = "".join(w if pattern.search(w) else f"{w} " for w in chars).strip()
    else:
        # Join with spaces
        s = " ".join(w for w in chars)
    # s = " ".join([w.strip() for w in chars if w.strip()])

    s = re.sub(
        r"\s+", " ", s
    )  # replace any successive whitespace characters with a space

    if simplified_chinese:  # convert to standard chinese
        s = to_simple(s)

    if simplified_japanese:
        s = simplified_japanese(s)

    if merge_single_char:
        s = merge_single_characters(s)

    if remove_erhua:
        s = remove_erhua_zh(s)

    if remove_fillers:
        for c in FILLER_CHARS:
            s = s.replace(c, "")

    # Restore protected special tokens.
    if placeholders:
        for ph, tok in placeholders.items():
            s = s.replace(ph, tok)

    return s


class MultilingualTextNorm:
    def __init__(
        self,
        case: str = None,
        remove_symbols: bool = False,
        remove_diacritics: bool = False,
        space_between_cjk: bool = False,
        simplified_chinese: bool = False,
        simplified_japanese: bool = False,
    ):
        self.case = case
        self.remove_symbols = remove_symbols
        self.remove_diacritics = remove_diacritics
        self.space_between_cjk = space_between_cjk
        self.simplified_chinese = simplified_chinese
        self.simplified_japanese = simplified_japanese

    def __call__(self, s):
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        if self.remove_symbols:
            s = remove_symbols_only(s)

        if self.remove_diacritics:
            s = remove_symbols_and_diacritics(s)

        if self.case == "lower":
            s = s.lower()
        elif self.case == "upper":
            s = s.upper()

        pattern = re.compile(
            r"([\u1100-\u11ff\u2e80-\ua4cf\ua840-\uD7AF\uF900-\uFAFF\uFE30-\uFE4F\uFF65-\uFFDC\U00020000-\U0002FFFF\u3000-\u303F\uff01-\uff60])"
        )
        chars = pattern.split(s)
        chars = [ch for ch in chars if ch.strip()]
        if not self.space_between_cjk:
            # Join with space only if the segment is not CJK
            s = "".join(w if pattern.search(w) else f"{w} " for w in chars).strip()
        else:
            # Join with spaces
            s = " ".join(w for w in chars)
        # s = " ".join([w.strip() for w in chars if w.strip()])

        s = re.sub(
            r"\s+", " ", s
        )  # replace any successive whitespace characters with a space

        if self.simplified_chinese:  # convert to standard chinese
            s = to_simple(s)

        if self.simplified_japanese:
            s = simplified_japanese(s)

        return s


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        help="input text file",
    )

    parser.add_argument("--output", type=Path, help="output text file")

    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help="english char case (i.e. lower or upper)",
    )

    parser.add_argument(
        "--remove-symbols",
        type=bool,
        default=False,
        help="whether to remove symbols",
    )

    parser.add_argument(
        "--remove-diacritics",
        type=bool,
        default=False,
        help="whether to remove diacritics",
    )

    parser.add_argument(
        "--space-between-cjk",
        type=bool,
        default=False,
        help="whether to add spaces between CJK chars",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    logging.basicConfig(level=logging.INFO)
    if args.output.is_file():
        logging.info(f"{args.output} already exists")
    else:
        logging.info(f"Doing text normalization and output to {args.output}")
        with open(args.output, "w") as f:
            for line in tqdm(open(args.input, "r").readlines()):
                line_tn = text_normalization(
                    line,
                    case=args.case,
                    remove_symbols=args.remove_symbols,
                    remove_diacritics=args.remove_diacritics,
                    space_between_cjk=args.space_between_cjk,
                )
                f.write(line_tn + "\n")
