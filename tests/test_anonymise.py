from trader.llm import anonymise


def test_names_and_tickers_become_numbered_placeholders_tagged_companies_first():
    text = "Apple Inc (NASDAQ:AAPL) rose while Microsoft's cloud grew; shares of AAPL and $MSFT gained."

    out, mapping = anonymise.anonymise(text, tagged=["MSFT", "AAPL"])

    assert (
        out
        == "Company 2 Inc (Company 2) rose while Company 1's cloud grew; shares of Company 2 and Company 1 gained."
    )
    assert mapping == {"Company 1": "MSFT", "Company 2": "AAPL"}


def test_untagged_companies_are_numbered_in_order_of_appearance_with_all_their_names():
    out, mapping = anonymise.anonymise("Nvidia supplies Meta Platforms, formerly Facebook (FB).", tagged=[])
    assert out == "Company 1 supplies Company 2, formerly Company 2 (Company 2)."
    assert mapping == {"Company 1": "NVDA", "Company 2": "META"}


def test_ordinary_words_and_bare_single_letters_are_left_alone():
    out, _ = anonymise.anonymise("An apple a day; Vitamin V; Visa (NYSE:V) and $V.", tagged=[])
    assert out == "An apple a day; Vitamin V; Company 1 (Company 1) and Company 1."


def test_dates_and_years_are_masked():
    text = "On March 3, 2024 the fiscal 2025 outlook (Q3 2024) was cut; filed 2024-03-04, 3/4/2024 and Mar. 5."
    out, _ = anonymise.anonymise(text, tagged=[])
    assert out == "On [DATE] the fiscal [YEAR] outlook (Q3 [YEAR]) was cut; filed [DATE], [DATE] and [DATE]."


def test_restore_maps_placeholders_back_and_keeps_anything_else():
    assert anonymise.restore(["Company 2", "Company 9", "XYZ"], {"Company 2": "AAPL"}) == [
        "AAPL",
        "Company 9",
        "XYZ",
    ]


def test_every_universe_name_is_known():
    from trader import config

    assert set(anonymise.COMPANY_NAMES) == set(config.UNIVERSE)
