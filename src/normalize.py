"""Stage N: name / address normalisation (IMPLEMENTATION_PLAN §3).

Two layers:
  * basic_* expressions: lower-case, Latin accent stripping, junk removal, tokenisation. Used by the
    dictionary miner (mine.py) and as the first step of the full normaliser.
  * normalize(): applies the mined dictionaries (Indic->Latin, address token canonicalisation, legal forms)
    and parses the fields used by features (legal forms, core name, primary house number, street, ...).

Hand-written knowledge is limited to generic regexes and the short legal-form / honorific seed lists below;
everything language- or region-specific (Indic transliteration, state codes, street-type abbreviations,
French abbreviations) is mined from the provided data by mine.py.
"""
import json

import polars as pl

from . import config

# ---- seed lists (documented in the methodology; generic, not business lookups) ----
LEGAL_SEED = {
    "llc": "llc", "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co", "ltd": "ltd", "limited": "ltd", "pvt": "pvt", "private": "pvt",
    "llp": "llp", "lp": "lp", "pc": "pc", "pllc": "pllc", "plc": "plc",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sa": "sa", "sci": "sci",
    "ei": "ei", "cie": "cie", "ets": "ets", "etablissements": "ets",
}
HONORIFICS = {"sri", "shri", "smt", "mr", "mrs", "ms", "dr", "the"}
ALIAS_RE = r"\s(?:dba|d/b/a|f/k/a|fka|a/k/a|aka|formerly|as)\s"
NULL_TOKENS_RE = r"(?i)<null>|\bnull\b|\bn/a\b"
UNIT_WORDS = r"(?:unit|apt|apartment|suite|ste|fl|floor|flr|pmb|po box|p o box|box|room|rm|bldg|building)"
# "n°" / "n °" (French numero; the degree sign survives accent stripping) was missing: 46k France test records had no
# house number and a street starting with "°" (found in the qualitative France audit; absent from train)
NUM_PREFIX_RE = r"^(?:#+|h\.?\s?no\.?|hn|door\s*no\.?|n\s*°\.?|no\.?|plot\s*no\.?|plot|kh\.?\s*no\.?|shop\s*no\.?|flat\s*no\.?)\s*"
# a parenthesised leading house number "(46) rue des hannetons" (20k France test records, 76 of 10M train records)
PAREN_NUM_RE = r"^\((\d+[a-z]{0,3})\)\s*"
INDIC_RE = r"[ऀ-෿]"


def strip_latin_accents(e: pl.Expr) -> pl.Expr:
    """NFKD, then drop combining marks that follow a Latin letter (keeps Indic vowel signs intact)."""
    return e.str.normalize("NFKD").str.replace_all(r"([A-Za-z])\p{Mn}+", "${1}")


def basic_name(col: str = "name") -> pl.Expr:
    e = strip_latin_accents(pl.col(col).fill_null("").str.to_lowercase())
    e = (e.str.replace_all(r"\(\s*id:[^)]*\)", " ")            # (ID: 123)
          .str.replace_all(r"\s#\d+\s*$", " ")                  # trailing #35632
          .str.replace_all(r"\s-\s*\d{6,}\s*$", " ")            # trailing phone-like number
          .str.replace_all(r"\bm/s\b", " ")                     # M/s honorific
          .str.replace_all(r"@", "a")
          .str.replace_all(r"&", " and ")
          .str.replace_all(r"\b([a-z])\.([a-z])\.(?:([a-z])\.?)?", "${1}${2}${3}")  # l.l.c. -> llc, p.c. -> pc
          .str.replace_all(r"[^\p{L}\p{M}\p{N}]+", " ")
          .str.replace_all(r"\s+", " ").str.strip_chars())
    return e


def basic_addr(col: str = "addr") -> pl.Expr:
    e = strip_latin_accents(pl.col(col).fill_null("").str.replace_all(NULL_TOKENS_RE, " ").str.to_lowercase())
    return e.str.replace_all(r"\s+", " ").str.strip_chars(" ,")


def tokens(e: pl.Expr) -> pl.Expr:
    return e.str.split(" ").list.eval(pl.element().filter(pl.element() != ""))


def _map_tokens(tok_list: pl.Expr, mapping: dict) -> pl.Expr:
    """Replace each token by mapping[token] if present (dictionary lookup inside list.eval)."""
    if not mapping:
        return tok_list
    return tok_list.list.eval(pl.element().replace(mapping))


def _dedupe_consecutive(tok_list: pl.Expr) -> pl.Expr:
    el = pl.element()
    return tok_list.list.eval(el.filter(el != el.shift(1).fill_null("\x00")))


def _deleet(tok_list: pl.Expr) -> pl.Expr:
    """Map digits to letters only inside tokens that mix letters and digits (not ordinals like 3rd)."""
    el = pl.element()
    mixed = el.str.contains(r"\p{L}") & el.str.contains(r"\d") & ~el.str.contains(r"^\d+(st|nd|rd|th)$")
    fixed = el.str.replace_all("0", "o").str.replace_all("1", "l").str.replace_all("3", "e").str.replace_all("5", "s")
    return tok_list.list.eval(pl.when(mixed).then(fixed).otherwise(el))


def load_dicts() -> dict:
    p = config.WORK_DIR / "dicts.json"
    return json.loads(p.read_text()) if p.exists() else {"indic": {}, "addr": {}, "legal": {}}


def normalize(df: pl.DataFrame, dicts: dict, country: str | None = None) -> pl.DataFrame:
    """df has idx, name, addr (one country). Returns the normalised columns keyed by idx.

    Address dictionaries are per country label (mined per country); a country without a mined
    dictionary simply gets no address canonicalisation (country is an open set)."""
    legal_map = {**LEGAL_SEED, **dicts.get("legal", {})}
    indic = dicts.get("indic", {})
    addr_map = dicts.get("addr", {}).get(country, {}) if country else {}
    comp_map = dicts.get("comp", {}).get(country, {}) if country else {}

    raw = df.select("idx", "name", "addr").with_columns(
        pl.col("name").str.contains(INDIC_RE).alias("was_indic"),
        pl.col("name").str.contains(r"^\s*(?:--|<<|>>|\*|#|~|\.\.)").alias("had_junk"),
        (pl.col("name").str.to_uppercase() == pl.col("name")).alias("upper_style"),
        pl.col("name").str.to_lowercase().str.contains(r"^[a-z0-9-]+\.(?:com|net|org|in|co\.in|fr|biz|io|us|co)$").alias("is_domain"),
    )
    # --- names ---
    n = raw.with_columns(basic_name("name").alias("_n"))
    n = n.with_columns(
        pl.col("_n").str.replace(ALIAS_RE + r".*$", "").alias("_primary"),
        pl.col("_n").str.extract(ALIAS_RE + r"(.*)$", 1).fill_null("").alias("alias"),
        pl.when(pl.col("is_domain")).then(pl.col("name").str.to_lowercase().str.extract(r"^([a-z0-9-]+)\.", 1))
          .otherwise(None).alias("domain_stem"),
    )
    # a domain-style name ("glyphifylaw.com") is compared through its stem
    n = n.with_columns(pl.when(pl.col("is_domain")).then(pl.col("domain_stem")).otherwise(pl.col("_primary")).alias("_primary"))
    tl = tokens(pl.col("_primary"))
    tl = _map_tokens(_deleet(tl), indic)
    n = n.with_columns(_dedupe_consecutive(tl).alias("_tok"))
    n = n.with_columns(
        # tokens still in Indic script after dictionary mapping = out-of-vocabulary
        pl.col("_tok").list.eval(pl.element().str.contains(INDIC_RE)).list.sum().alias("indic_oov"),
    )
    n = n.with_columns(
        pl.col("_tok").list.eval(pl.element().replace_strict(legal_map, default=None).drop_nulls())
          .list.unique().list.sort().list.join(" ").alias("legal"),
        pl.col("_tok").list.eval(pl.element().filter(~pl.element().is_in(list(legal_map)) &
                                                     ~pl.element().is_in(list(HONORIFICS)))).alias("_core"),
        pl.col("_tok").list.eval(pl.element().is_in(list(HONORIFICS))).list.any().alias("had_honorific"),
    )
    n = n.with_columns(
        pl.col("_tok").list.join(" ").alias("name_clean"),
        pl.col("_core").list.join(" ").alias("name_core"),
        pl.col("_core").list.join("").alias("name_nospace"),
        pl.col("_core").list.eval(pl.element().str.slice(0, 1)).list.join("").alias("acronym"),
        pl.col("_core").list.len().alias("n_core_tok"),
    )
    # --- addresses ---
    a = raw.select("idx", "addr").with_columns(basic_addr("addr").alias("_a"))
    comps = pl.col("_a").str.split(",").list.eval(pl.element().str.strip_chars(" .")).list.eval(
        pl.element().filter(pl.element() != ""))
    a = a.with_columns(comps.alias("_comps"))
    # whole-component canonicalisation first (states: 'tn' / 'tamil nadu' / Indic script; mined in mine.py)
    if comp_map:
        a = a.with_columns(pl.col("_comps").list.eval(pl.element().replace(comp_map)).alias("_comps"))
    # canonicalise every token inside components with the mined address map
    a = a.with_columns(pl.col("_comps").list.eval(
        pl.element().str.split(" ").list.eval(pl.element().replace(addr_map) if addr_map else pl.element())
        .list.join(" ")).alias("_comps"))
    # secondary (unit / floor / po box) components, primary component = first non-secondary with a digit
    unit_re = rf"^(?:{UNIT_WORDS})\b"
    a = a.with_columns(
        pl.col("_comps").list.eval(pl.element().filter(pl.element().str.contains(unit_re))).alias("_sec"),
        pl.col("_comps").list.eval(pl.element().filter(~pl.element().str.contains(unit_re) &
                                                       pl.element().str.contains(r"\d"))).list.first().alias("_prim"),
    )
    a = a.with_columns(pl.col("_prim").fill_null("").str.replace(NUM_PREFIX_RE, "").str.replace(PAREN_NUM_RE, "${1} ")
                         .alias("_prim"))
    a = a.with_columns(
        pl.col("_prim").str.extract(r"^(\d+[a-z]{0,3}(?:\s*[-/]\s*\w+)*)", 1).alias("_pnum"),
    )
    suffix_re = r"^\d+\s*(bis|ter|quater|[a-z])\b"
    a = a.with_columns(
        pl.col("_pnum").str.extract(r"^(\d+)", 1).str.slice(0, 9).cast(pl.Int64, strict=False).alias("pno"),
        pl.col("_prim").str.extract(suffix_re, 1).alias("psuffix"),
        pl.col("_pnum").str.extract_all(r"\d+").list.join("-").alias("ptuple"),
        pl.col("_prim").str.replace(r"^\d+[a-z]{0,3}(?:\s*[-/]\s*\w+)*", "")
          .str.replace(r"^\s*(?:bis|ter|quater|[a-z])\b", "").str.strip_chars(" -/").alias("street"),
        pl.col("_sec").list.join(" ").str.extract_all(r"\d+").list.join("-").alias("sec_nos"),
        pl.col("_a").str.extract_all(r"\d+").list.unique().list.join(" ").alias("all_nums"),
        pl.col("_comps").list.len().alias("n_comps"),
        (pl.col("_a") == "").alias("addr_missing"),
        pl.col("_comps").list.join(", ").alias("addr_clean"),
        pl.col("_comps").list.last().alias("addr_last"),
        pl.col("_comps").list.first().alias("addr_first"),
    )
    out = n.select("idx", "name_clean", "name_core", "name_nospace", "acronym", "legal", "alias", "domain_stem",
                   "is_domain", "was_indic", "indic_oov", "had_junk", "had_honorific", "upper_style", "n_core_tok")
    out = out.join(a.select("idx", "addr_clean", "pno", "psuffix", "ptuple", "street", "sec_nos", "all_nums",
                            "n_comps", "addr_missing", "addr_first", "addr_last"), on="idx")
    return out


def run() -> None:
    dicts = load_dicts()
    for split in config.SPLITS:
        rec = pl.read_parquet(config.split_dir(split) / "records.parquet")
        # score one representative per duplicate group; S1 rows are their own representatives
        reps = rec.filter(pl.col("idx") == pl.col("rep"))
        out = pl.concat([normalize(reps.filter(pl.col("country") == c), dicts, c)
                         for c in reps["country"].unique().sort().to_list()])
        out.write_parquet(config.split_dir(split) / "norm.parquet")
        print(f"[normalize] {split}: {out.height:,} rows")


if __name__ == "__main__":
    run()
