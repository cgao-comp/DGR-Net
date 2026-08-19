import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer,BertConfig
import os
from pathlib import Path

__all__ = ['BertTextEncoder']

TRANSFORMERS_MAP = {
    'bert': (BertModel, BertTokenizer),
    'roberta': (RobertaModel, RobertaTokenizer),
}

class BertTextEncoder(nn.Module):
    def __init__(self, use_finetune=False, transformers='bert', pretrained='bert-base-uncased'):
        super().__init__()
        tokenizer_class = TRANSFORMERS_MAP[transformers][1]
        model_class = TRANSFORMERS_MAP[transformers][0]

        # The pre-4.x Hugging Face cache layout used by the original project
        # may not exist on a new machine.  Only pass a local path when it is
        # real; otherwise pass the official model id so transformers can use
        # its current cache layout or download it.
        legacy_cache = Path.home() / '.cache' / 'huggingface' / 'transformers' / pretrained
        hub_snapshots = Path.home() / '.cache' / 'huggingface' / 'hub' / f"models--{pretrained.replace('/', '--')}" / 'snapshots'
        snapshot_dirs = sorted((path for path in hub_snapshots.glob('*') if path.is_dir()), key=lambda path: path.stat().st_mtime, reverse=True) if hub_snapshots.is_dir() else []
        cached_source = legacy_cache if legacy_cache.is_dir() else (snapshot_dirs[0] if snapshot_dirs else None)
        model_source = os.environ.get('DGR_BERT_PATH') or str(cached_source or pretrained)
        local_only = os.path.isdir(model_source)
        try:
            self.model = model_class.from_pretrained(model_source, local_files_only=local_only, from_tf=False)
            self.tokenizer = tokenizer_class.from_pretrained(model_source, local_files_only=local_only)
        except OSError as error:
            if not local_only:
                raise OSError(
                    "Unable to download BERT and no complete local cache was found. "
                    "Set DGR_BERT_PATH to a local bert-base-uncased model directory, or pre-download it with "
                    "`huggingface-cli download bert-base-uncased --local-dir /path/to/bert-base-uncased`."
                ) from error
            raise
        self.use_finetune = use_finetune
    
    def get_tokenizer(self):
        return self.tokenizer
    
    # def from_text(self, text):
    #     """
    #     text: raw data
    #     """
    #     input_ids = self.get_id(text)
    #     with torch.no_grad():
    #         last_hidden_states = self.model(input_ids)[0]  # Models outputs are now tuples
    #     return last_hidden_states.squeeze()
    
    def forward(self, text):
        """
        text: (batch_size, 3, seq_len)
        3: input_ids, input_mask, segment_ids
        input_ids: input_ids,
        input_mask: attention_mask,
        segment_ids: token_type_ids
        """
        input_ids, input_mask, segment_ids = text[:,0,:].long(), text[:,1,:].float(), text[:,2,:].long()
        if self.use_finetune:
            last_hidden_states = self.model(input_ids=input_ids,
                                            attention_mask=input_mask,
                                            token_type_ids=segment_ids)[0]  # Models outputs are now tuples
        else:
            with torch.no_grad():
                last_hidden_states = self.model(input_ids=input_ids,
                                                attention_mask=input_mask,
                                                token_type_ids=segment_ids)[0]  # Models outputs are now tuples
        return last_hidden_states
