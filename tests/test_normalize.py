import polars as pl
from src.normalize import basic_name, normalize

DICTS = {"indic": {"एस": "ace", "इंडस्ट्रीज": "industries", "प्राइवेट": "private", "लिमिटेड": "limited"},
         "legal": {}, "comp": {"US": {"tennessee": "tn"}}, "addr": {"US": {"ln": "lane", "ct": "court"}}}


def test_basic_name_cleaning():
    got = pl.DataFrame({"name": ["BRYAN, BONITA GREEN L.L.C.", "M/s Ansuya & Sons Pvt. Ltd.", "Léarning (ID: 123) Co #35632"]}) \
        .select(basic_name()).to_series().to_list()
    assert got == ["bryan bonita green llc", "ansuya and sons pvt ltd", "learning co"]


def test_normalize_us():
    df = pl.DataFrame({"idx": [0, 1], "name": ["Knoxville Urology [Associates]", "Harb0r Brokerage Inc"],
                       "addr": ["Tennessee, Knoxville, 308 Laurel Ridge LN", "Unit Apartment 323, Fargo, #4817 28th Avenue"]})
    out = normalize(df, DICTS, "US").sort("idx")
    assert out["name_core"].to_list() == ["knoxville urology associates", "harbor brokerage"]
    assert out["legal"].to_list() == ["", "inc"]
    assert out["pno"].to_list() == [308, 4817]
    assert out["street"].to_list()[0] == "laurel ridge lane"
    assert out["sec_nos"].to_list()[1] == "323"
    assert out["addr_first"].to_list()[0] == "tn"


def test_normalize_indic_and_tuple():
    df = pl.DataFrame({"idx": [0], "name": ["एस इंडस्ट्रीज प्राइवेट लिमिटेड"], "addr": ["H.no. 9-17-158 Hashmi Colony, Nizamabad"]})
    out = normalize(df, DICTS, "India")
    assert out["name_core"][0] == "ace industries" and out["legal"][0] == "ltd pvt"
    assert out["pno"][0] == 9 and out["ptuple"][0] == "9-17-158" and out["indic_oov"][0] == 0


def test_normalize_france_suffix():
    df = pl.DataFrame({"idx": [0], "name": ["Club Privée SARL"], "addr": ["15 bis Chemin du Moulin, NULL, Lille"]})
    out = normalize(df, {"indic": {}, "legal": {}}, "France")
    assert out["pno"][0] == 15 and out["psuffix"][0] == "bis" and out["street"][0] == "chemin du moulin"
    assert out["addr_clean"][0] == "15 bis chemin du moulin, lille" and out["legal"][0] == "sarl"
