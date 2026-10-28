# text-classification ERNIE 文本分类
## 功能说明
加载训练完成的 ERNIE 文本分类模型，对输入文本进行分类，输出各类别的预测概率和预测标签索引。

## 使用方法
```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 加载模型
path = "./ernie3-checkpoints-save"
device = torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForSequenceClassification.from_pretrained(path).to(device)
model.eval()

# 输入文本
text = "中国羽毛球队的小型庆功会就此拉开帷幕"
inputs = tokenizer(text, return_tensors="pt")
input_ids=inputs["input_ids"].to(device)
attention_mask=inputs["attention_mask"].to(device)

# 预测
outputs = model(input_ids=input_ids, attention_mask=attention_mask)

print("预测概率：", torch.softmax(outputs.logits, dim=-1))
print("预测标签索引：", torch.argmax(outputs.logits, dim=-1).item())
```

## 输出说明
- **预测概率**：文本属于各类别的概率。
- **预测标签索引**：概率最高的类别编号，具体类别需根据训练时的标签映射确定。
