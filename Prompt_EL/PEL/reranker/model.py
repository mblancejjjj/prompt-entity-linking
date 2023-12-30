from torch import nn, optim
from transformers import BertModel, AutoModel

class EntityRankerClassifier(nn.Module):
    def __init__(self, n_classes, PRE_TRAINED_MODEL_NAME):
        super(EntityRankerClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        # 将输入传递给BERT模型，得到输出的池化表示
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        # 对池化表示进行Dropout
        output = self.drop(pooled_output)
        # 将Dropout后的表示传递给线性层
        linear_output = self.out(output)
        # 使用Softmax激活函数，将模型的输出转换为概率分布
        probabilities = nn.functional.sigmoid(linear_output)
        return probabilities
    #def forward(self, input_ids, attention_mask):
     #   _, pooled_output = self.bert(
      #    input_ids=input_ids,
       #   attention_mask=attention_mask,
        #  return_dict=False
        #)
        #output = self.drop(pooled_output)
        #return self.out(output)



