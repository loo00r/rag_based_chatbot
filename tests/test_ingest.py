from ingest import is_section, parse_docs

SAMPLE_HTML = """<html><body>
<p>1. ЗАГАЛЬНІ ПОЛОЖЕННЯ</p>
<p>1.1. Ці Правила визначають порядок дорожнього руху.</p>
<p>1.2. У цих Правилах терміни вживаються у такому значенні.</p>
<p>2. ЗАГАЛЬНІ ОБОВ'ЯЗКИ ВОДІЇВ</p>
<p>2.1. Водій зобов'язаний мати при собі документи.</p>
</body></html>"""


def test_is_section_true():
    assert is_section("7. ПРІОРИТЕТ РУХУ") is True
    assert is_section("1. ЗАГАЛЬНІ ПОЛОЖЕННЯ") is True
    assert is_section("12. ЗУПИНКА І СТОЯНКА") is True


def test_is_section_false():
    assert is_section("7.1. Водій зобов'язаний...") is False
    assert is_section("звичайний текст") is False
    assert is_section("") is False
    assert is_section("1. не верхній регістр") is False


def test_parse_docs_count():
    docs = parse_docs(SAMPLE_HTML)
    assert len(docs) >= 3


def test_parse_docs_metadata():
    docs = parse_docs(SAMPLE_HTML)
    for d in docs:
        assert "rule_id" in d.metadata
        assert "section" in d.metadata


def test_parse_docs_rule_ids():
    docs = parse_docs(SAMPLE_HTML)
    ids = [d.metadata["rule_id"] for d in docs]
    assert "1.1" in ids
    assert "1.2" in ids
    assert "2.1" in ids


def test_parse_docs_section_assigned():
    docs = parse_docs(SAMPLE_HTML)
    rule_21 = next(d for d in docs if d.metadata["rule_id"] == "2.1")
    assert "ОБОВ" in rule_21.metadata["section"]
