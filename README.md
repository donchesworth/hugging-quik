# hugging-quik
A helper package to make loading/unloading huggingface models quik-er

This was originally part of [pytorch-quik](https://github.com/donchesworth/pytorch-quik), but its requirements are large, and hugging-quik is a simple way to pull and use huggingface.co pretrained models and tokenizers.


hugging-quik does the following:
    - get_tokenizer: pull a tokenizer from the huggingface suite of models
    - save_tokenizer: save the tokenizer to disk
    - get_encodings: use the tokenizer to get text encodings
    - get_pretrained_model: pull a model from the huggingface suite
    - save_pretrained_model: save the model to disk
