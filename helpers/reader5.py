# coding: utf-8
import json
from nltk.tokenize import word_tokenize
import torch
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np

import pandas as pd
import os
from os.path import join as pjoin
from helpers.path_util import from_project_root, dirname
import helpers.json_util as ju

from tqdm import tqdm
import pickle

# from helpers.lm_embeddings import data2ids, list2str
import sys
# sys.path.append('gen_emb/')
# from gen_emb.albert_emb import PreEmbeddedLM
# from gen_emb.distilbert_emb import list2str

from transformers import DistilBertTokenizer
import nltk
#LIAR-PLUS
ROOT_PROJ_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/dataset/LIAR-RAW"
# DATA_FILE_PATH = "..\\dataset\\oracles"#six-class
# PRETRAINED_URL = from_project_root("dataset/embedding/glove.840B.300d.word2vec.vocab")
VOCAB_EMB_URL = 'embeddings.npy'##generated from PRETRAINED
VOCAB_URL = "vocab.json"
CHAR_VOCAB_URL = "char_vocab.json"

# MAX_ORACLE= 55 # 最多的oracle的句子数

global_max_claimnum = 2

if 'pub' in ROOT_PROJ_PATH:
    LABEL_IDS = {"false": 0, "true": 1, "mixture": 2, "unproven": 3}
elif 'fever' in ROOT_PROJ_PATH:
    LABEL_IDS = {"REFUTES": 0, "SUPPORTS": 1}
elif 'RAWFC' in ROOT_PROJ_PATH or 'seefact' in ROOT_PROJ_PATH:
    LABEL_IDS = {"false": 0, "true": 1, "half": 2}
elif 'LIAR-RAW' in ROOT_PROJ_PATH:
    LABEL_IDS = {"pants-fire": 0, "false": 1, "barely-true": 2, "half-true": 3, "mostly-true": 4, "true": 5}
else:
    LABEL_IDS = {"pants-fire": 0, "false": 1, "barely-true": 2, "half-true": 3, "mostly-true": 4, "true": 5}

TOP_K_ORACLE_NUMS = 5#prepare 5 FOR PUB_ORACLE , 5 FOR LIAR-PLUS

class myDataset(Dataset):
    def __init__(self, data_file, report_each_claim=30):
        super().__init__()
        self.df = self.read_from_dir(data_file)
        
        self.finetune = False#False
        self.report_each_claim = report_each_claim

        self._len = len(self.df)        

        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')#("bert-base-uncased")
        
        self.event_id, self.claim, self.label, \
            self.explain, self.report_links, self.report_contents, \
                self.report_domains, self.tok_sents, self.tok_sent_ids = self.load_raw(self.df)

    def __getitem__(self, index):
        return self.event_id[index], self.claim[index], self.label[index], \
            self.explain[index], self.report_links[index], self.report_contents[index], \
                self.report_domains[index], self.tok_sents[index], self.tok_sent_ids[index]

    def __len__(self):
        return self._len

    def read_from_dir(self, path):
        ''''1 csv pd.read_csv
            2 json return all filenames in dir'''

        if os.path.isfile(path) and 'LIAR-RAW' in path:
            with open(path, 'r', encoding='utf-8') as f:
                all_data = json.load(f) # all_data is a list contain all data.
            return all_data
        else: 
            # 得到文件夹下所有json文件的名称
            filenames = os.listdir(path)
            name_list = []
            for name in filenames:
                if '.json' in name: 
                    name_list.append(name)

            # 读取所有json
            if len(name_list) == 1:
                all_data = ''
                # for fever, only allow 1 file in {mode} dir 
                for file in name_list:
                    filename = pjoin(path, file)# root/xxxx.json
                    with open(filename, 'r', encoding='utf-8') as json_file:
                        all_data = json.load(json_file)
            else:
                all_data = []
                # for our snope, allow many files in {mode} dir
                for file in name_list:
                    filename = pjoin(path, file)# root/xxxx.json
                    with open(filename, 'r', encoding='utf-8') as json_file:
                        obj = json.load(json_file)
                        all_data.append(obj)
            return all_data

    def load_raw(self, df):
        '''parsing dict objs to list '''
        raw_data = [[] for _ in range(9)]##event_id, claim, label, explain, (link, content, domain, report_sents, report_is_evidence) 0/1      
        for obj in tqdm(df):
            report_tok_sents = []           
            report_tok_sent_ids = []  
       
            report_links = []
            report_contents = []
            report_domains = []  
            event_id, claim, label, explain, reports = obj['event_id'], obj['claim'], obj['label'], obj['explain'], obj['reports']

            raw_data[0].append(event_id)
            raw_data[1].append(claim)
            raw_data[2].append(label)
            raw_data[3].append(explain)
            for s in reports[:self.report_each_claim]: # 截取
                report_links.append(s['link']) 
                report_contents.append(s['content']) #全文, doc
                report_domains.append(s['domain']) 
                # for tokenized sents
                tok_sents = [] 
                tok_sent_ids = []   
                for ts in s['tokenized']:
                    tok_sents.append(ts['sent'])
                    tok_sent_ids.append(ts['is_evidence'])
                    
                report_tok_sents.append(tok_sents)
                report_tok_sent_ids.append(tok_sent_ids)
            raw_data[4].append(report_links) 
            raw_data[5].append(report_contents) #全文, doc sent_list
            raw_data[6].append(report_domains) 

            raw_data[7].append(report_tok_sents) 
            raw_data[8].append(report_tok_sent_ids) 

        return raw_data

    def my_collate(self, batch):
        '''collect data with your style'''
        event_id, claim, label, \
            explain, link, content, \
                domain, report_sents, report_is_evidence = [[] for _ in range(9)]
        raw_data_list = []

        raw_text_dict = {}
        lm_ids_dict = {}

        lm_embs_dict = {}

        num_report_docs = []
        for i,item in enumerate(batch):
            event_id.append(item[0])
            claim.append(item[1])
            label.append(item[2])

            explain.append(item[3])
            link.append(item[4])
            content.append(item[5])
            # save number of reports
            num_report_docs.append(len(item[5]))

            domain.append(item[6])
            report_sents.append(item[7])
            report_is_evidence.append(item[8])

        raw_data_list.append(event_id)
        raw_data_list.append(claim)
        raw_data_list.append(label)
        raw_data_list.append(explain)
        raw_data_list.append(link)
        raw_data_list.append(content)
        raw_data_list.append(domain)
        raw_data_list.append(report_sents)
        raw_data_list.append(report_is_evidence)

        lm_ids_dict['num_report_docs'] = num_report_docs

        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # claim label
        label_ids = gen_label_id(raw_data_list[2], device=_device)
        # needing mask
        oracle_ids = gen_oracle_id(raw_data_list[-1], device=_device)

        # num of each report
        num_oracle_list = [torch.sum(nums, dim=1) for nums in oracle_ids]
        num_oracle_list.insert(0, torch.zeros(self.report_each_claim))
        num_oracle_eachdoc = pad_sequence(num_oracle_list, batch_first=True)
        num_oracle_eachdoc = num_oracle_eachdoc[1:]
        # (batch_size, )

        # Task: doc label : whehter to select this report for evidence
        lm_ids_dict['label_ids'] = label_ids

        lm_ids_dict['gold_doc_labels'] = [[int(bool(sum(doc))) for doc in claim_docs] for claim_docs in raw_data_list[8]]
        lm_ids_dict['gold_doc_masks'] = torch.FloatTensor([[True]*t + [False]*(num_oracle_eachdoc.shape[1] - t) for t in num_report_docs]).to(_device)

        raw_text_dict['_CLAIM_TOK'] = raw_data_list[1]
        raw_text_dict['_TGT_TOK'] = raw_data_list[3]
        raw_text_dict['_SRC_TOK'] = raw_data_list[7]

        lm_ids_dict['claim_ids'] = []
        lm_ids_dict['claim_masks'] = []
        lm_ids_dict['src_ids'] = []
        lm_ids_dict['src_masks'] = []
        
        lm_embs_dict['claim_embs'] = []
        lm_embs_dict['src_embs'] = []

        lm_ids_dict['src_sent_num'] = [[len(reports) for reports in claims] for claims in raw_data_list[7]]        
        lm_ids_dict['num_oracle_eachdoc'] = num_oracle_eachdoc
        lm_ids_dict['report_domains'] = gen_domain_id(raw_data_list[6], device=_device)

        max_length = cal_max_word_num(raw_data_list)
        max_length = max_length if max_length < 300 else 300

        for i in range(len(raw_data_list[1])):
            encoded_claim = self.tokenizer(raw_data_list[1][i], return_tensors='pt', padding=True, truncation=True, max_length=max_length) 
            claim_ids, claim_attention_mask = encoded_claim['input_ids'], encoded_claim['attention_mask']

            lm_ids_dict['claim_ids'].append(claim_ids)#repr
            lm_ids_dict['claim_masks'].append(claim_attention_mask)#repr

            encoded_src_docs = [self.tokenizer(doc, return_tensors='pt', padding=True, truncation=True, max_length=max_length) for doc in raw_data_list[7][i]]
            src_ids, src_attention_mask = [], []
            for doc_dict in encoded_src_docs:
                src_ids.append(doc_dict['input_ids'])
                src_attention_mask.append(doc_dict['attention_mask'])
            
            lm_ids_dict['src_ids'].append(src_ids)
            lm_ids_dict['src_masks'].append(src_attention_mask)

        return oracle_ids, label_ids, raw_text_dict, lm_ids_dict


    def data2ids(self, sentences, labels=None, max_length=None, add_special_tokens=False):
        input_ids,attention_mask=[],[]
        for i in range(len(sentences)):
            encoded_dict = self.tokenizer.encode_plus(
            sentences[i],
            add_special_tokens = add_special_tokens,      
            max_length = max_length,           
            pad_to_max_length = True,
            truncation=True,
            return_tensors = 'pt',         
            )
            input_ids.append(encoded_dict['input_ids'])
            attention_mask.append(encoded_dict['attention_mask'])

        input_ids = torch.cat(input_ids, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)

        input_ids = torch.LongTensor(input_ids)
        attention_mask = torch.LongTensor(attention_mask)
            
        return input_ids, attention_mask

def mergelist(sentlist):
    if type(sentlist[0]) == list:
        ret = []
        for item in sentlist:
            ret.extend(item)
        return [ret]

def len_allsents(_CLAIM_TOK):
    '''@_CLAIM_TOK：list of list'''
    lens = []
    for sents in _CLAIM_TOK:
        _len = 0
        for s in sents:
            _len += len(s)#maybe more than one sents
        lens.append(_len)
    assert len(lens) == len(_CLAIM_TOK)
    return lens

def gen_oracle_id(_ORACLE_IDS, device):
    '''select oracle ids from ruling_tokenized'''
    doc_ids = [[torch.LongTensor(id).to(device) for id in doc_ids] for doc_ids in _ORACLE_IDS]
    oracle_ids = [pad_sequence(_ids, batch_first=True) for _ids in doc_ids]
    return oracle_ids 


# =========================================================================
# 👇 核心修复逻辑1：动态识别标签字典，并配置防崩溃安全网
# =========================================================================
def gen_label_id(_LABEL, device, label_ids=None):
    '''generate label ids for claims with auto dataset recognition'''
    global LABEL_IDS
    # 动态获取主程序中可能已被替换的全局字典，否则使用自带字典
    current_label_ids = label_ids if label_ids is not None else LABEL_IDS
    
    try:
        # 尝试使用当前的字典去解析标签
        _ids = [current_label_ids[la] for la in _LABEL]
    except KeyError:
        # 💡 终极兜底方案：如果解析失败（比如 6 分类字典遇到了 RAWFC 独有的 'half' 标签），直接在此处强行切为 3 分类字典！
        fallback_label_ids = {"false": 0, "half": 1, "true": 2}
        _ids = [fallback_label_ids[la] for la in _LABEL]
        
    id_tensors = torch.LongTensor(_ids).to(device)
    return id_tensors
# =========================================================================


def get_idx_list(len_list):
    ''' generate real index of the tensor.'''
    rets = []
    _tmp = 0
    for value in len_list:
        _tmp = _tmp + value
        rets.append(_tmp)
    return rets


# =========================================================================
# 👇 核心修复逻辑2：针对 RAWFC 等缺失字典的情况进行防撞墙保护
# =========================================================================
def gen_domain_id(domain, data_url=ROOT_PROJ_PATH, filename='vocab_article_source.json', device='cuda'):
    '''obtain domain ids'''
    source_url = pjoin(data_url, filename)### domain
    
    # 🌟 新增防撞墙保护：如果找不到领域字典，直接返回全 0 的占位符，不影响特征提取
    if not os.path.exists(source_url):
        return [torch.zeros(len(ds), dtype=torch.long).to(device) for ds in domain]
        
    source_vocab = ju.load(source_url)
    report_domains = []
    unk_idx = 1
    for ds in domain:
        # domain item to item id
        ds_item = torch.LongTensor([source_vocab[item] if item in source_vocab else unk_idx
                                        for item in ds]).to(device)
        report_domains.append(ds_item)

    return report_domains
# =========================================================================


def cal_max_word_num(raw_data):
    ''' return max len'''
    
    event_id, claim, label,  explain, link, content,  domain, report_sents, report_is_evidence = raw_data
    _CLAIM_TOK = []
    _SRC_TOK = []
    for cl, rs in zip(claim, report_sents):
        _CLAIM_TOK.append(nltk.word_tokenize(cl))
        _SRC_TOK.append([[nltk.word_tokenize(sent) for sent in r] for r in rs])

    cla_len = [len(cla) for cla in _CLAIM_TOK]

    src_len = []
    src_tok_list = []
    for tok in _SRC_TOK:
        for t in tok:
            src_tok_list.extend(t)
            src_len.extend([len(s) for s in t])
    
    return max(cla_len+src_len)

    
def gen_sent_tensors(raw_data, device='auto', data_url=ROOT_PROJ_PATH, npy_path=VOCAB_EMB_URL):
    '''generate input tensors for 1 batch'''
    event_id, claim, label,  explain, link, content,  domain, report_sents, report_is_evidence = raw_data
    _CLAIM_TOK = []
    _SRC_TOK = []
    for cl, rs in zip(claim, report_sents):
        _CLAIM_TOK.append(nltk.word_tokenize(cl))
        _SRC_TOK.append([[nltk.word_tokenize(sent) for sent in r] for r in rs])

    npy_path_url = pjoin(data_url, npy_path)
    vocab_url = pjoin(data_url, VOCAB_URL)
    char_vocab_url = pjoin(data_url, CHAR_VOCAB_URL)
    source_url = pjoin(data_url, 'vocab_article_source.json')

    vocab = ju.load(vocab_url)
    char_vocab = ju.load(char_vocab_url)
    source_vocab = ju.load(source_url)

    pretrained_emb = np.load(npy_path_url)

    sentences = list()
    sentence_words = list()
    sentence_word_lengths = list()
    sentence_word_indices = list()

    claim_tok_list = []

    report_domains = []
    for ds in domain:
        ds_item = torch.LongTensor([source_vocab[item] if item in source_vocab else unk_idx
                                        for item in ds]).to(device)
        report_domains.append(ds_item)
    report_domains = pad_sequence(report_domains, batch_first=True)

    for sents in _CLAIM_TOK:
        if not isinstance(sents[0], list): 
            claim_tok_list = [tok for tok in _CLAIM_TOK] 
            break

        _sent = []
        for s in sents:
            _sent = _sent + s
            _sent = _sent[:512]
        claim_tok_list.append([_sent])

    src_sent_num = [[len(t) for t in tok] for tok in _SRC_TOK]
    src_doc_num = [sum(nums) for nums in src_sent_num] 

    src_len = []
    src_tok_list = []
    for tok in _SRC_TOK:
        for t in tok:
            src_tok_list.extend(t)
            src_len.extend([len(s) for s in t])


    claim_tok_list = [[item] for item in claim_tok_list]
    src_tok_list = [[item] for item in src_tok_list]
    tok_list = claim_tok_list + src_tok_list 

    claim_len = [len(tok) for tok in claim_tok_list]
    num_claim = sum([len(tok) for tok in claim_tok_list])
    num_src = sum(src_doc_num)
    total_num = num_claim + num_src

    c_sentences_nums = claim_len
    s_sentences_nums = src_sent_num

    unk_idx = 1
    for sents in tok_list:
        for sent in sents:

            sentence = torch.LongTensor([vocab[word] if word in vocab else unk_idx
                                            for word in sent]).to(device)

            words = list()
            for word in sent:
                words.append([char_vocab[ch] if ch in char_vocab else unk_idx
                                for ch in word])

            word_lengths = torch.LongTensor([len(word) for word in words]).to(device)

            word_lengths, word_indices = torch.sort(word_lengths, descending=True)

            word_indices = word_indices.to(device)
            words = [torch.LongTensor(word).to(device) for word in words]

            words = pad_sequence(words, batch_first=True).to(device)

            sentences.append(sentence)
            sentence_words.append(words)
            sentence_word_lengths.append(word_lengths)
            sentence_word_indices.append(word_indices)

    sentence_lengths = [len(sentence) for sentence in sentences]
    sentences = pad_sequence(sentences, batch_first=True)

    c_sentences             = sentences[:num_claim]
    c_sentence_lengths      = sentence_lengths[:num_claim]
    c_sentence_words        = sentence_words[:num_claim]
    c_sentence_word_lengths = sentence_word_lengths[:num_claim]
    c_sentence_word_indices = sentence_word_indices[:num_claim]
    c_results = (c_sentences, c_sentence_lengths, c_sentence_words, c_sentence_word_lengths, c_sentence_word_indices, c_sentences_nums)


    s_sentences             = sentences[num_claim:total_num]
    s_sentence_lengths      = sentence_lengths[num_claim:total_num]
    s_sentence_words        = sentence_words[num_claim:total_num]
    s_sentence_word_lengths = sentence_word_lengths[num_claim:total_num]
    s_sentence_word_indices = sentence_word_indices[num_claim:total_num]
    s_results = (s_sentences, s_sentence_lengths, s_sentence_words, s_sentence_word_lengths, s_sentence_word_indices, s_sentences_nums)

    t_results = None
    return c_results, t_results, s_results, claim_tok_list, src_tok_list, src_sent_num, report_domains

def read_from_dir(path):
    ''''return all filenames in dir'''
    filenames = os.listdir(path)
    name_list = []
    for name in filenames:
        if '.json' in name: 
            name_list.append(name)
    return name_list


def load_df_dataset(mode, lm_emb, filepath=ROOT_PROJ_PATH, batch_size=64, n_workers=2, shuffle=False):
    '''mode = train, test, val'''
    root_path = pjoin(filepath, f'{mode}')

    dataset = myDataset(root_path, lm_emb)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=dataset.my_collate, num_workers=n_workers)
    return train_loader

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    from distilbert_emb import DistilEmbeddings, list2str
    lm_emb = DistilEmbeddings()
    data_loader = load_df_dataset("test", lm_emb)
    for idx, (claim_tensors, just_tensors, src_tensors, oracle_ids, label_ids) in enumerate(data_loader):
        print(idx, claim_tensors, just_tensors, src_tensors, oracle_ids, label_ids)