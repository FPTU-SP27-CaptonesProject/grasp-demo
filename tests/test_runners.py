from grasp_demo.runners import run_column


class DummyServices:
    def __init__(self):
        self.health = {"reader": {"ready": True}, "retriever": {"ready": True}, "proxy": {"ready": True}, "strong": {"ready": False, "reason": "no key"}}
        self.config = {"ircot": {}, "grasp": {}}

    def enabled(self, column):
        return (False, "no key") if column == "rag_235b" else (True, None)


def test_disabled_column_is_rejected_without_starting_a_runner():
    try:
        run_column(DummyServices(), "rag_235b", {"id": "q", "question": "Q", "metadata": {}}, lambda *_: None)
    except RuntimeError as error:
        assert "no key" in str(error)
    else:
        raise AssertionError("expected disabled column")
