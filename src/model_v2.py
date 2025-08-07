from typing import Optional

from torch import Tensor
from transformers import CLIPModel,BertConfig
from transformers.models.bert.modeling_bert import BertLayer
import torch.nn as nn
import torch
import torch.nn.functional as F
import copy


from backbone import build_backbone

from transformers import BertTokenizer, BertModel


class RCLMuFN(nn.Module):
    def __init__(self, args):
        super(RCLMuFN, self).__init__()
        self.model = CLIPModel.from_pretrained("/home/xiongjie/data/models/clip-vit-base-patch32")
        self.text_linear =  nn.Sequential(
            nn.Linear(args.text_size, args.image_size),
            nn.Dropout(args.dropout_rate),
            nn.GELU()
        )
        self.image_linear =  nn.Sequential(
            nn.Linear(args.image_size, args.image_size),
            nn.Dropout(args.dropout_rate),
            nn.GELU()
        )
        self.classifier_fuse = nn.Linear(args.image_size , args.label_number)
        self.loss_fct = nn.CrossEntropyLoss()


    def forward(self, inputs, batch, labels):
        output = self.model(**inputs,output_attentions=True)

        # 提取对应的特征
        text_features = output['text_model_output']['last_hidden_state']  # 128，77，512
        image_features = output['vision_model_output']['last_hidden_state']  # 128，50，768
        text_feature = output['text_model_output']['pooler_output'] # 64，512
        image_feature = output['vision_model_output']['pooler_output'] # 64，768
        text_feature = self.text_linear(text_feature)  # 64，768
        image_feature = self.image_linear(image_feature)  # 64,768

        fuse_feature = text_feature + image_feature
        output = fuse_feature

        # Predict
        logits_fuse = self.classifier_fuse(output)  # 64,2  output
        fuse_score = nn.functional.softmax(logits_fuse, dim=-1)  # 64,2

        score = fuse_score

        outputs = (score,) # (64,2)
        if labels is not None:
            loss_fuse = self.loss_fct(logits_fuse, labels)
            loss = loss_fuse
            outputs = (loss,) + outputs
        return outputs

