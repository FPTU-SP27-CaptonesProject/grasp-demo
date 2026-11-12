import pytest

from grasp_demo import services as services_module
from grasp_demo.services import DemoServices, load_proxy

ADAPTERS = {"sft": {"path": "", "label": "SFT"}, "grpo": {"path": "", "label": "GRPO s100"}}


def ok(name):
    def loader(services):
        setattr(services, name, object())
    return loader


def fail(message):
    def loader(services):
        raise RuntimeError(message)
    return loader


class Proxy:
    def __init__(self, names):
        self.names = names

    def close(self):
        pass


def config():
    return {"proxy": {"adapters": {k: dict(v) for k, v in ADAPTERS.items()}}}


def test_reader_down_disables_every_reader_column():
    services = DemoServices.from_config(config(), loaders={"reader": fail("reader down"), "retriever": ok("retriever"),
                                                           "proxy": lambda s: setattr(s, "proxy", Proxy({"sft": "default", "grpo": "grpo"})), "strong": ok("strong_reader")})
    assert services.enabled("rag_235b") == (True, None)
    for column in ("rag_7b", "ircot_7b", "grasp_sft", "grasp_grpo"):
        enabled, reason = services.enabled(column)
        assert not enabled and "reader down" in reason
    assert services.adapter_metadata["grpo"]["label"] == "GRPO s100"


def test_missing_bedrock_key_disables_only_the_235b_column():
    services = DemoServices.from_config(config(), loaders={"reader": ok("reader"), "retriever": ok("retriever"),
                                                           "proxy": lambda s: setattr(s, "proxy", Proxy({"sft": "default", "grpo": "grpo"})), "strong": fail("BEDROCK_API_KEY is not set")})
    assert services.enabled("rag_235b")[0] is False
    assert all(services.enabled(c)[0] for c in ("rag_7b", "ircot_7b", "grasp_sft", "grasp_grpo"))


def test_missing_grpo_adapter_disables_only_its_column(tmp_path, monkeypatch):
    sft = tmp_path / "sft"
    sft.mkdir()
    (sft / "adapter_config.json").write_text("{}", encoding="utf-8")
    values = config()
    values["proxy"]["adapters"]["sft"]["path"] = str(sft)
    values["proxy"]["adapters"]["grpo"]["path"] = str(tmp_path / "nope")
    monkeypatch.setattr(services_module, "HFProxy", lambda adapters: Proxy({name: name for name in adapters}))
    services = DemoServices.from_config(values, loaders={"reader": ok("reader"), "retriever": ok("retriever"), "proxy": load_proxy, "strong": ok("strong_reader")})
    assert services.enabled("grasp_sft") == (True, None)
    enabled, reason = services.enabled("grasp_grpo")
    assert not enabled and "nope" in reason


def test_vllm_backend_is_chosen_from_config(tmp_path, monkeypatch):
    sft = tmp_path / "sft"
    sft.mkdir()
    (sft / "adapter_config.json").write_text("{}", encoding="utf-8")
    seen = {}
    monkeypatch.setenv("READER_API_KEY", "k")
    monkeypatch.setattr(services_module, "VLLMProxy", lambda url, key, adapters: seen.update(url=url, key=key, names=list(adapters)) or Proxy({"sft": "demo-sft"}))
    values = {"proxy": {"backend": "vllm", "url": "http://localhost:8001/v1", "adapters": {"sft": {"path": str(sft), "label": "SFT"}}}}
    services = DemoServices(values)
    load_proxy(services)
    assert seen == {"url": "http://localhost:8001/v1", "key": "k", "names": ["sft"]}


def test_unknown_backend_is_an_error(tmp_path):
    sft = tmp_path / "sft"
    sft.mkdir()
    (sft / "adapter_config.json").write_text("{}", encoding="utf-8")
    services = DemoServices({"proxy": {"backend": "tgi", "adapters": {"sft": {"path": str(sft), "label": "SFT"}}}})
    with pytest.raises(ValueError, match="tgi"):
        load_proxy(services)
