from histogram import histogram_from_sims


def test_histogram_basic_stats_and_shape():
    sims = [0.9, 0.85, 0.88, 0.92, 0.1, 0.12, 0.15, 0.05, 0.3, 0.2]
    result = histogram_from_sims(sims)

    assert result["n"] == 10
    assert 15 <= result["buckets_requested"] <= 60
    assert len(result["buckets"]) == result["buckets_requested"]
    assert sum(b["count"] for b in result["buckets"]) == 10

    stats = result["stats"]
    assert stats["min"] <= stats["mean"] <= stats["max"]
    assert stats["median"] is not None
    assert isinstance(stats["looks_normal"], bool)

    assert result["above_mean_plus_2std"] >= 0
    assert result["above_mean_plus_3std"] <= result["above_mean_plus_2std"]
    assert result["mean_plus_2std"] == stats["mean"] + 2 * stats["std"]


def test_histogram_fixed_bucket_count_is_respected():
    sims = [0.1 * i for i in range(1, 21)]
    result = histogram_from_sims(sims, buckets=10)

    assert result["buckets_requested"] == 10
    assert len(result["buckets"]) == 10


def test_histogram_handles_few_points_without_crashing():
    result = histogram_from_sims([0.5, 0.6, 0.7])

    assert result["n"] == 3
    assert result["stats"]["looks_normal"] is False
    assert result["stats"]["normaltest_p"] is None


def test_histogram_handles_constant_scores_without_crashing():
    import json

    result = histogram_from_sims([0.5] * 12)

    assert result["n"] == 12
    assert result["stats"]["std"] == 0
    # skewness/excess_kurtosis are NaN for constant data; must come back as None
    # (JSON-serializable), not NaN.
    assert result["stats"]["skewness"] is None
    assert result["stats"]["excess_kurtosis"] is None
    assert sum(b["count"] for b in result["buckets"]) == 12
    json.dumps(result)  # must not raise
