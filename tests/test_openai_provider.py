"""Tests for batching and response ordering without a network call."""

from types import SimpleNamespace

from mrm_assistant.embeddings.openai_provider import embed_texts


class FakeEmbeddingsAPI:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def create(self, *, model: str, input: list[str], encoding_format: str) -> SimpleNamespace:
        assert model == "test-model"
        assert encoding_format == "float"
        self.calls.append(input)
        data = [
            SimpleNamespace(index=index, embedding=[float(index), float(len(text))])
            for index, text in reversed(list(enumerate(input)))
        ]
        return SimpleNamespace(data=data)


def test_embed_texts_batches_and_restores_provider_order() -> None:
    fake_api = FakeEmbeddingsAPI()
    fake_client = SimpleNamespace(embeddings=fake_api)
    texts = [f"text {number}" for number in range(51)]

    vectors = embed_texts(fake_client, texts, "test-model")

    assert [len(batch) for batch in fake_api.calls] == [50, 1]
    assert vectors[0] == [0.0, 6.0]
    assert vectors[-1] == [0.0, 7.0]
