from rag_labor_code.parser import parse

SAMPLE = """ЧАСТЬ ПЕРВАЯ
Раздел I. ОБЩИЕ ПОЛОЖЕНИЯ
Глава 1. ОСНОВНЫЕ НАЧАЛА
Статья 1. Цели
Первый абзац.
пункт; (в ред. Федерального закона от 02.07.2013 N 185-ФЗ)
Статья 2. Принципы
Текст."""


def test_parse_sample():
    arts = parse(SAMPLE)
    assert [a.number for a in arts] == ["1", "2"]
    assert arts[0].paragraphs == ["Первый абзац.", "пункт;"]
    assert arts[0].chapter.startswith("Глава 1")


def test_hyphen_number_and_repealed():
    text = (
        "ЧАСТЬ ПЕРВАЯ\n"
        "Статья 348.11-1. Основания\nТекст.\n"
        "Статья 85. - Утратила силу.\n"
        "Статья 86. Другая\nТекст."
    )
    arts = parse(text)
    assert [a.number for a in arts] == ["348.11-1", "86"]


def test_editorial_note_dropped():
    text = (
        "ЧАСТЬ ПЕРВАЯ\n"
        "Статья 81. Расторжение\n"
        "Текст.\n"
        "Действие части первой статьи 81 (в редакции Федерального закона от 07.10.2022 N 376-ФЗ) "
        "распространяется на правоотношения, возникшие с 21.09.2022.\n"
        "Еще текст."
    )
    assert parse(text)[0].paragraphs == ["Текст.", "Еще текст."]


def test_signature_dropped():
    text = (
        "ЧАСТЬ ПЕРВАЯ\n"
        "Статья 424. Заключительная\n"
        "Текст статьи.\n"
        "Президент\nРоссийской Федерации\nВ. ПУТИН\nМосква, Кремль\n"
        "30 декабря 2001 года\nN 197-ФЗ"
    )
    assert parse(text)[-1].paragraphs == ["Текст статьи."]