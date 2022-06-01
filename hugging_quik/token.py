import numpy as np
from transformers import (
    BertTokenizer,
    RobertaTokenizer,
    MarianTokenizer,
    BatchEncoding,
    PreTrainedTokenizer,
)
from typing import Optional, List, Union
import shutil
from pathlib import Path
import json
from hugging_quik.io import json_write
from hugging_quik.model import HUGGING_MODELS
import sys
import logging
from functools import partial

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

Tokenizers = Union[BertTokenizer, RobertaTokenizer]

BATCH_ENCODE_KWARGS = {
    "add_special_tokens": True,
    "return_attention_mask": True,
    "pad_to_max_length": True,
    "max_length": 256,  # max is 512
    "truncation": True,
    "return_tensors": "pt",
}


def get_tokenizer(
    model_type: Optional[str] = "bert",
    cache_dir: Optional[str] = None,
    kwargs: Optional[dict] = {},
) -> Tokenizers:
    """Get a BERT, RoBERTa, or MarianMT tokenizer from transformers to create
    encodings.

    Args:
        model_type (str, optional): The type of tokenizer ("bert", "roberta",
        "marianmt"). Defaults to "bert".
        cache_dir (str, optional): The directory to cache the tokenizer. This
        helps save_tokenizer find it.
        kwargs (dict, optional): If a MarianMT mode, the source and target
        languages

    Returns:
        Tokenizers: The tokenizer to be used by get_encodings.
    """
    model_dict = HUGGING_MODELS[model_type]
    model_name = model_dict["model-name"]
    if isinstance(model_name, partial):
        model_name = model_name(**kwargs)
    for _ in range(5):
        while True:
            try:
                tokenizer = model_dict["tokenizer"].from_pretrained(
                    pretrained_model_name_or_path=model_name,
                    do_lower_case=True,
                    cache_dir=cache_dir,
                )
            except ValueError:
                logger.info("Connection error, trying again")
                continue
            break
    return tokenizer


def update_tokenizer_path(serve_path):
    configpath = Path(serve_path).joinpath("tokenizer_config.json")
    newtokenizer = "./tokenizer.json"
    with open(configpath) as f:
        tconfig = json.load(f)
    logger.info(tconfig)
    srcpath = Path(tconfig["tokenizer_file"])
    dstpath = serve_path.joinpath(newtokenizer)
    shutil.copyfile(srcpath, dstpath)
    tconfig["tokenizer_file"] = newtokenizer
    json_write(serve_path, configpath, tconfig)


def save_tokenizer(tokenizer: Tokenizers, serve_path: Path):
    """Serving a torch model requires saving the tokenizer, and
    it needs to be in the path that the model archive is built.

    Args:
        tokenizer (Tokenizers): The BERT or RoBERTa tokenizer used
        serve_path (Path): The location for building the model archive
    """
    tokenizer.save_pretrained(serve_path)
    if not isinstance(tokenizer, MarianTokenizer):
        update_tokenizer_path(serve_path)


def get_encodings(
    array_list: List[np.ndarray],
    idx: int,
    bert_type: Optional[str] = "bert",
    tokenizer: Optional[PreTrainedTokenizer] = None,
    kwargs: Optional[dict] = {},
) -> List[BatchEncoding]:
    """Get BERT or RoBERTa encodings for a list of np.ndarrays for training,
    testing (and validation).

    Args:
        array_list (List[np.ndarray]): The list of training, test, (and
        validation) arrays.
        idx (int): The column index of the text in the original pd.DataFrame
        bert_type (str, optional): The type of bert model to be used. Defaults
        to "bert".
        tokenizer (PreTrainedTokenizer, optional): The tokenizer to use
        kwargs (dict, optional): For overriding BATCH_ENCODE_KWARGS

    Returns:
        List[BatchEncoding]: A list of training, test (and validation)
        encodings of text.
    """
    BATCH_ENCODE_KWARGS.update(kwargs)
    logger.info(f"kwargs are now: {BATCH_ENCODE_KWARGS['return_tensors']}")
    if tokenizer is None:
        tokenizer = get_tokenizer(bert_type)
    encode_list = [
        tokenizer(list(array[:, idx]), **BATCH_ENCODE_KWARGS)
        for array in array_list
    ]
    return encode_list, tokenizer
