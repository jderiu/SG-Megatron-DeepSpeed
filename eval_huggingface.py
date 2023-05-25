import json, os, torch

from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers.models.bloom.tokenization_bloom_fast import BloomTokenizerFast
from transformers import StoppingCriteria, StoppingCriteriaList


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = [stop.to(device) for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False

bloom_model = "bigscience/bloom-1b1"
path = '../Theory of Evaluation/llm_train/models/oass1/test2/checkpoint-12300'
path = '/mnt/e/work_space_data/bloom_chkp/bloom-1b1/hf'
# path = 'bigscience/bloom-560m'
device = 'cpu'
tokenizer = BloomTokenizerFast.from_pretrained(bloom_model)
model = BloomForCausalLM.from_pretrained(path)
#model.to(device)

p_tag = '<|prompter|>:'
a_tag = '<|assistant|>:'
print('Loaded Pipeline')

stop_words_ids = [tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze() for stop_word in [p_tag]]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

prompt = input("Enter propmpt: ")
text = f'{p_tag} {prompt}\n{a_tag}'
while not prompt == 'exit':
    inputs = tokenizer(text, return_tensors="pt").to(device=device)

    with torch.cuda.amp.autocast(), torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=0.7, num_return_sequences=1, stopping_criteria=stopping_criteria)
    otext = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    if otext.endswith(p_tag):
        otext = otext[:-len(p_tag)]
    print(otext)
    prompt = input("Enter propmpt: ")
    if prompt == 'clear':
        text = ''
        prompt = input("Enter propmpt: ")
    text = f'{otext}\n{p_tag} {prompt}\n{a_tag}'
