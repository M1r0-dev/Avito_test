import numpy as np

from avito_retrieval.filters import requested_min_rating
from avito_retrieval.bm25 import SparseBM25
from avito_retrieval.fusion import reciprocal_rank_fusion, unique_top_k
from avito_retrieval.metrics import recall_at_k
from avito_retrieval.language import dominant_script
from avito_retrieval.statistics import paired_recall_test, wilson_interval


def test_unique_and_rrf() -> None:
    assert unique_top_k(["a", "a", "b"], 2) == ["a", "b"]
    assert reciprocal_rank_fusion([["a", "b"], ["b", "c"]], top_k=3)[0] in {"a", "b"}


def test_recall() -> None:
    assert recall_at_k({"q": ["a", "x"]}, {"q": {"a", "b"}}, 2) == 0.5


def test_bm25_reusable_score_vector() -> None:
    index = SparseBM25(min_df=1).fit(["ремонт телефона", "ремонт автомобиля", "маникюр"])
    rows, scores = index.search("ремонт", top_k=2)
    full = index.score("ремонт")
    assert set(rows) == {0, 1}
    assert np.allclose(scores, full[rows])


def test_bm25_inverted_index_is_bit_identical_to_matmul() -> None:
    documents = ["ремонт телефона и ноутбука", "ремонт автомобиля", "маникюр и педикюр",
                 "ремонт квартиры под ключ", "педикюр на дому", "ноутбук ремонт экрана"]
    index = SparseBM25(min_df=1).fit(documents)
    for query in ["ремонт ноутбука", "педикюр ремонт маникюр", "неизвестное слово", "", "ремонт ремонт"]:
        assert np.array_equal(index.score(query), index.score_matmul(query))
    assert index.score("ремонт").dtype == np.float32


def test_rating_parser() -> None:
    assert requested_min_rating("Рейтинг пользователя 4 звезды и выше") == 4.0
    assert requested_min_rating("") is None


def test_script_detector() -> None:
    assert dominant_script("ремонт iphone") == "mixed"
    assert dominant_script("ремонт айфона") == "cyrillic"
    assert dominant_script("iphone repair") == "latin"


def test_paired_statistics() -> None:
    truth = {"q1": {"a"}, "q2": {"b"}}
    better = {"q1": ["a"], "q2": ["b"]}
    worse = {"q1": ["x"], "q2": ["b"]}
    result = paired_recall_test(better, worse, truth, n_resamples=200, seed=1)
    assert result.mean_delta == 0.5
    stricter = paired_recall_test(
        better, worse, truth, n_resamples=200, seed=1, confidence=0.975
    )
    assert stricter.ci_low <= result.ci_low
    assert stricter.ci_high >= result.ci_high
    low, high = wilson_interval(95, 100)
    assert low < 0.95 < high
