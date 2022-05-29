import numpy as np
from torch import Tensor
from transformers import (
    BertTokenizer,
    RobertaTokenizer,
    MarianTokenizer,
    BertForSequenceClassification,
    RobertaForSequenceClassification,
    MarianMTModel,
    PreTrainedModel,
    logging as tlog,
)
from typing import Optional, Union, Dict, OrderedDict
from pathlib import Path
from . import utils
from argparse import Namespace
from torch.nn.parallel import DistributedDataParallel as DDP
import sys
import logging
from functools import partial

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

Tokenizers = Union[BertTokenizer, RobertaTokenizer]
Models = Union[BertForSequenceClassification, RobertaForSequenceClassification]

# ESEN = {"src": "es", "tgt": "en"}
HUGGING_MODELS = {
    "bert": {
        "model-name": "bert-base-uncased",
        "tokenizer": BertTokenizer,
        "model-head": BertForSequenceClassification,
        "mode": "sequence_classification",
    },
    "roberta": {
        "model-name": "roberta-base",
        "tokenizer": RobertaTokenizer,
        "model-head": RobertaForSequenceClassification,
        "mode": "sequence_classification",
    },
    "marianmt": {
        "model-name": partial("Helsinki-NLP/opus-mt-{source}-{target}".format),
        "tokenizer": MarianTokenizer,
        "model-head": MarianMTModel,
        "mode": "sequence_to_sequence",
    },
}


def model_info(
    model_type: Optional[str] = "bert",
    labels: Optional[Tensor] = None,
    kwargs: Optional[dict] = None,
) -> Dict[str, Union[str, bool]]:

    """Create a serving config dictionary for model serving.

    Args:
        model_type (str, optional): The type of the model to be used ("bert",
        "roberta", or "marianmt"). Defaults to "bert".
        labels (Tensor, optional): A tensor of the labels in the dataset. This
        will affect the size of the network final layer
        kwargs (dict, optional): If the model-name is a partial (e.g. MarianMT)
        this will provide the missing text. Default is español to english.

    Returns:
        Dict [str, Any]: A dict that could be provided for torch serve
    """
    model_dict = HUGGING_MODELS[model_type]
    model_name = model_dict["model-name"]
    if isinstance(model_name, partial):
        model_name = model_name(**kwargs)
    serve_config = {
        "model_name": model_name,
        "mode": model_dict["mode"],
        "do_lower_case": True,
        "save_mode": "pretrained",
        "max_length": 256,
        "captum_explanation": False,
        "embedding_name": model_type,
    }
    if labels:
        serve_config["num_labels"] = str(len(np.unique(labels)))
    return serve_config


def get_pretrained_model(
    model_type: Optional[str] = "bert",
    labels: Optional[Tensor] = None,
    kwargs: Optional[dict] = {},
) -> Models:
    """Get a pretrained model from transformers.

    Args:
        labels (Tensor, optional): A tensor of the labels in the dataset. This
        will affect the size of the network final layer
        model_type (str, optional): The type of the model to be used ("bert",
        "roberta", or "marianmt"). Defaults to "bert".

    Returns:
        Models: The pretrained model to be used in training.
    """
    if labels is not None:
        num_lbls = len(np.unique(labels))
    model_dict = HUGGING_MODELS[model_type]
    model_name = model_dict["model-name"]
    if model_type in ["bert", "roberta"]:
        kwargs = {
            "num_labels": num_lbls,
            "output_attentions": False,
            "output_hidden_states": False,
        }
    else:
        if isinstance(model_name, partial):
            model_name = model_name(**kwargs)
            kwargs = {}
    tlog.set_verbosity_error()
    for i in range(0, 3):
        while True:
            try:
                model = model_dict["model-head"].from_pretrained(
                    model_name, **kwargs
                )
            except ValueError:
                logger.info("Connection error, trying again")
                continue
            break
    return model


def save_pretrained_model(
    model: PreTrainedModel,
    args: Optional[Namespace] = None,
    best_epoch: Optional[int] = None,
    serve_path: Optional[Path] = None,
    state_dict: Optional[OrderedDict] = None,
):
    """Serving a torch model requires saving the model, and
    it needs to be in the path that the model archive is built.

    Args:
        model (PreTrainedModel): The BERT or RoBERTa model (for this
        purpose post-training)
        args (Namespace): The list of arguments for building paths
        best_epoch (int): The epoch from which to pull the state_dict
        serve_path (Path, optional): Directory to store files for building
        the model archive. Defaults to None.
        state_dict (OrderedDict, optional) Model state dictionary. Could be
        model.state_dict() if using default
    """
    if isinstance(model, DDP):
        model = model.module
    if serve_path is None:
        serve_path = Path.cwd()
    if state_dict is None:
        state_dict = model.state_dict()
    if "module.classifier.bias" in state_dict:
        logger.info("Removing distribution from model")
        state_dict = utils.consume_prefix_in_state_dict_if_present(
            state_dict, "module."
        )
    model.save_pretrained(save_directory=serve_path, state_dict=state_dict)
