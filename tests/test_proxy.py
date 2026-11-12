from grasp_grpo.model import Action

from grasp_demo.proxy import HFProxy, VLLMProxy

ADAPTERS = {"sft": {"path": "/ck/sft", "label": "SFT"}, "grpo": {"path": "/ck/s100", "label": "GRPO"}}


class FakeModel:
    def __init__(self):
        self.active_adapter, self.loaded, self.seen = "default", [], []

    def load_adapter(self, path, adapter_name, is_trainable):
        self.loaded.append((path, adapter_name, is_trainable))

    def set_adapter(self, name):
        self.active_adapter = name

    def parameters(self):
        return []


def test_hf_proxy_loads_the_first_adapter_as_base_and_switches():
    model = FakeModel()

    def generate(m, tokenizer, prompt, **kwargs):
        m.seen.append((m.active_adapter, kwargs["temperature"], kwargs["samples"]))
        return [Action("Yes", [], [])]

    proxy = HFProxy(ADAPTERS, load=lambda path, trainable: (model, "tok", "base"), generate=generate)
    assert proxy.names == {"sft": "default", "grpo": "grpo"}
    assert model.loaded == [("/ck/s100", "grpo", False)]
    proxy.generate_fn("grpo")("rd", "prompt", 1)
    proxy.generate_fn("sft")("rd", "prompt", 1)
    assert model.seen == [("grpo", 0.0, 1), ("default", 0.0, 1)]
    assert model.active_adapter == "default"


class FakeAdmin:
    def __init__(self, fail_first_load=False):
        self.calls, self.fail_first_load = [], fail_first_load

    def post(self, endpoint, payload):
        self.calls.append((endpoint, payload["lora_name"]))
        if self.fail_first_load and endpoint == "load_lora_adapter" and len(self.calls) == 1:
            raise RuntimeError("already loaded")


class FakePolicy:
    def __init__(self, served):
        self.served = served

    def generate(self, prompt, *, samples, max_new_tokens, temperature, top_p):
        return [Action(f"{self.served}:{temperature}", [], [])]


def test_vllm_proxy_loads_named_adapters_and_unloads_on_close():
    admin = FakeAdmin()
    proxy = VLLMProxy("http://x/v1", "key", ADAPTERS, admin=admin, policy_factory=lambda served, path: FakePolicy(served))
    assert proxy.names == {"sft": "demo-sft", "grpo": "demo-grpo"}
    assert proxy.generate_fn("grpo")("qf", "p", 1)[0].text == "demo-grpo:0.0"
    proxy.close()
    assert admin.calls == [("load_lora_adapter", "demo-sft"), ("load_lora_adapter", "demo-grpo"),
                           ("unload_lora_adapter", "demo-sft"), ("unload_lora_adapter", "demo-grpo")]


def test_vllm_proxy_reloads_a_name_left_by_an_old_server():
    admin = FakeAdmin(fail_first_load=True)
    VLLMProxy("http://x/v1", "key", {"sft": ADAPTERS["sft"]}, admin=admin, policy_factory=lambda served, path: FakePolicy(served))
    assert admin.calls == [("load_lora_adapter", "demo-sft"), ("unload_lora_adapter", "demo-sft"), ("load_lora_adapter", "demo-sft")]
