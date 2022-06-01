import re
from hugging_quik.model import (
    HUGGING_MODELS,
    get_pretrained_model,
    model_info,
    save_pretrained_model,
)
from hugging_quik.token import (
    get_encodings,
    get_tokenizer,
    save_tokenizer,
)
from pathlib import Path
from transformers import BatchEncoding, logging as tlog
src_tgt = {'source': 'es', 'target': 'en'}


def test_model_info(sample_labels):
    for model_type in HUGGING_MODELS.keys():
        serve_config = model_info(model_type, kwargs=src_tgt)
        assert isinstance(serve_config, dict)


def test_get_and_save_tokenizer():
    tlog.set_verbosity_error()
    for model_type, values in HUGGING_MODELS.items():
        tokenizer = get_tokenizer(model_type, kwargs=src_tgt)
        assert isinstance(tokenizer, values["tokenizer"])
        data_path = Path.cwd()
        save_tokenizer(tokenizer, data_path)


def test_get_encodings(sample_data):
    tlog.set_verbosity_error()
    for model_type, values in HUGGING_MODELS.items():
        tokenizer = get_tokenizer(model_type, kwargs=src_tgt)
        encode_list, tokenizer = get_encodings(
            [sample_data.values], 0, model_type, tokenizer)
        assert isinstance(tokenizer, values["tokenizer"])
        assert isinstance(encode_list[0], BatchEncoding)


def test_get_and_save_model(sample_labels):
    tlog.set_verbosity_error()
    for model_type, values in HUGGING_MODELS.items():
        model = get_pretrained_model(
            labels=sample_labels,
            model_type=model_type,
            kwargs=src_tgt,
        )
        assert isinstance(model, values["model-head"])
        save_pretrained_model(model)
