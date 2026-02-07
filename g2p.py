import re
import json
import numpy as np
import onnxruntime as ort
import os

MASTER = os.path.dirname(os.path.abspath(__file__))

TAGS = [
    "<PAD>",
    "VERB",
    "VBP",
    "VBN",
    "VBD",
    "ADJ",
    "ADV",
    "DT",
    "NOUN",
    "DEFAULT",
    "NONE"
]

PUNCS = r';:,.!?—…"()“”'

def split_text(text):
    alnum = r'a-zA-Z0-9'
    
    filter_pattern = f'[^{alnum}{PUNCS}\s]'
    cleaned_text = re.sub(filter_pattern, '', text)
    
    pattern = (
        rf'[{alnum}]+(?:[{PUNCS}][{alnum}]+)+'  # 内部含符号的单词
        rf'|[{alnum}]+'                             # 普通单词
        rf'|[{PUNCS}]'                          # 单个符号
    )
    
    tokens = re.findall(pattern, cleaned_text)
    return tokens

def number_to_words(num):
    ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", 
            "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", 
            "seventeen", "eighteen", "nineteen"]
    
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    
    scales = ["", "thousand", "million", "billion", "trillion"]

    def read_digit_by_digit(n_str):
        digit_map = {
            '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
            '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
        }
        return " ".join([digit_map[d] for d in n_str])

    def process_chunk(n):
        parts = []
        h = n // 100
        r = n % 100
        if h > 0:
            parts.append(ones[h])
            parts.append("hundred")
            if r > 0:
                parts.append("and")
        if r > 0:
            if r < 20:
                parts.append(ones[r])
            else:
                t = r // 10
                o = r % 10
                parts.append(tens[t])
                if o > 0:
                    parts.append(ones[o])
        return " ".join(parts)

    num_str = str(num)
    
    prefix = ""
    if num_str.startswith('-'):
        prefix = "minus "
        num_str = num_str[1:]

    if '.' in num_str:
        int_str, dec_str = num_str.split('.')
    else:
        int_str, dec_str = num_str, ""

    int_val = int(int_str) if int_str else 0
    int_words = ""

    if int_val == 0:
        int_words = "zero"
    elif int_val >= 10**15:
        int_words = read_digit_by_digit(int_str)
    else:
        chunks_data = []
        scale_index = 0
        temp_num = int_val
        while temp_num > 0:
            chunk_val = temp_num % 1000
            if chunk_val > 0:
                chunks_data.append({
                    'val': chunk_val,
                    'text': process_chunk(chunk_val),
                    'scale_idx': scale_index
                })
            temp_num //= 1000
            scale_index += 1

        chunks_data.reverse()
        res_list = []
        for i, data in enumerate(chunks_data):
            if len(res_list) > 0 and data['scale_idx'] == 0 and data['val'] < 100:
                res_list.append("and")
            res_list.append(data['text'])
            if scales[data['scale_idx']]:
                res_list.append(scales[data['scale_idx']])
        int_words = " ".join(res_list)

    if dec_str:
        decimal_words = "point " + read_digit_by_digit(dec_str)
        return f"{prefix}{int_words} {decimal_words}".strip()
    
    return f"{prefix}{int_words}".strip()

def is_number(s):
    try:
        float(s)  # 尝试转换成浮点数
        return True
    except ValueError:
        return False

def text_normalize(text):
    text = text.lower()
    text = split_text(text)
    expanded = []
    for word in text:
        if (is_number(word)):
            numbers = split_text(number_to_words(word))
            for num in numbers:
                expanded.append(num)
        else:
            expanded.append(word)
    return expanded

class XPOSAlternative:
    def __init__(self):
        tag_to_idx = {tag: i for i, tag in enumerate(TAGS)}
        n = len(TAGS)
        
        dist = [[float('inf')] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0.0

        v = 0.35
        raw_similarities = [
            ("NOUN", "VERB", 0.25),
            ("NOUN", "ADJ", 0.75),
            ("NOUN", "DEFAULT", v),
            ("VERB", "VBN", 1.0),
            ("VERB", "VBP", 1.0),
            ("VERB", "VBD", 1.0), 
            ("VERB", "ADJ", 0.25),
            ("VERB", "ADV", 0.5),
            ("VERB", "DEFAULT", v),
            ("VBN", "VBP", 0.75),
            ("VBN", "VBD", 1.25),
            ("VBN", "ADJ", 0.75),
            ("VBN", "DEFAULT", v),
            ("VBP", "VBD", 0.75),
            ("VBP", "DEFAULT", v),
            ("VBD", "DEFAULT", v),
            ("ADJ", "ADV", 0.5),
            ("ADJ", "DT", 0.75),
            ("ADJ", "DEFAULT", v),
            ("ADV", "DT", 0.25),
            ("ADV", "DEFAULT", v),
            ("DEFAULT", "DT", v),
            ("DEFAULT", "NONE", 0.5)
        ]

        for t1, t2, sim in raw_similarities:
            if sim > 0:
                d = 1.0 / sim
                u, v_idx = tag_to_idx[t1], tag_to_idx[t2]
                dist[u][v_idx] = dist[v_idx][u] = d

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]

        self._dist_map = {
            TAGS[i]: {TAGS[j]: dist[i][j] for j in range(n)}
            for i in range(n)
        }

    def find(self, predicted, valid_set):
        predicted = predicted.upper()
        
        if predicted in valid_set:
            return predicted
        
        if predicted not in self._dist_map:
            return next(iter(valid_set)) if valid_set else None

        best_tag = None
        min_dist = float('inf')
        
        scores = self._dist_map[predicted]
        
        for tag in valid_set:
            tag_upper = tag.upper()
            if tag_upper in scores:
                d = scores[tag_upper]
                if d < min_dist:
                    min_dist = d
                    best_tag = tag
        
        return best_tag if best_tag else (next(iter(valid_set)) if valid_set else None)

class G2P():
    def __init__(self):
        with open(f"{MASTER}/assets/us_gold_m.json", "r", encoding = "utf-8") as f:
            self.lexicon = json.load(f)
        
        with open(f"{MASTER}/assets/word2idx.json", "r", encoding = "utf-8") as f:
            self.word2idx = json.load(f)
        
        with open(f"{MASTER}/assets/char2idx.json", "r", encoding = "utf-8") as f:
            self.char2idx = json.load(f)
        
        with open(f"{MASTER}/assets/idx2tag.json", "r", encoding = "utf-8") as f:
            self.idx2tag = json.load(f)
        
        with open(f"{MASTER}/assets/idx2ipa.json", "r", encoding = "utf-8") as f:
            self.idx2ipa = json.load(f)
        
        self.etagger = ort.InferenceSession(f"{MASTER}/models/etagger.onnx", providers=["CPUExecutionProvider"]) # max seq length = 750
        self.etoddler = ort.InferenceSession(f"{MASTER}/models/etoddler.onnx", providers=["CPUExecutionProvider"]) # max output length = 120
        self.xpos_finder = XPOSAlternative()
        self.preran_oov = {}
    
    def __call__(self, text):
        text = text_normalize(text)
        # 1: <UNK>
        sentence_tokens = [self.word2idx.get(word, 1) for word in text]
        actual_sentence_length = len(sentence_tokens)
        if actual_sentence_length > 750:
            raise RuntimeError(f"too long. {len(sentence_tokens)} tokens")
        sentence_tokens = np.array((sentence_tokens + [0] * (750 - actual_sentence_length)), dtype = np.int64)

        ort_inputs = {self.etagger.get_inputs()[0].name: np.expand_dims(sentence_tokens, 0)}
        preds = np.array(self.etagger.run(None, ort_inputs))[0, 0, :actual_sentence_length, :]
        tags = np.argmax(preds, 1)

        ipas = []
        for word, tag in zip(text, tags):
            iv = self.lexicon.get(word, None)
            if iv == None:
                # OOV
                if word in PUNCS:
                    ipas.append(word)
                elif self.preran_oov.get(word, None) != None:
                    ipas.append(self.preran_oov[word])
                else:
                    # print(word)
                    char_tokens = [self.char2idx["<SOS>"]] + [self.char2idx.get(char, self.char2idx["-"]) for char in word] + [self.char2idx["<EOS>"]]
                    actual_chars_length = len(char_tokens)
                    if actual_chars_length > 120:
                        raise RuntimeError(f"too long. {len(actual_chars_length)} tokens")
                    char_tokens = np.array((char_tokens + [0] * (750 - actual_chars_length)), dtype = np.int64)

                    ort_inputs = {self.etoddler.get_inputs()[0].name: np.expand_dims(char_tokens, 0)}
                    preds = np.array(self.etoddler.run(None, ort_inputs))[0, 0]
                    result = ""
                    for i in preds:
                        if i == 2:
                            break
                        result += self.idx2ipa[str(i)]
                    
                    self.preran_oov[word] = result

                    ipas.append(result)
                    
            elif type(iv) == dict:
                possible_tags = [k for k, v in iv.items()]
                final_tag = self.xpos_finder.find(self.idx2tag[str(tag)], possible_tags)

                ipas.append(iv[final_tag])
            else:
                ipas.append(iv)
        
        return ipas

