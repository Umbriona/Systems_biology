from transformers import BertForMaskedLM, BertTokenizer, pipeline, AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
import numpy as np

import transformers
import time

transformers.logging.set_verbosity_error()

class bertModels:
    def __init__(self, topK, modelName, device, batch_size):
        self.topK = topK
        self.modelName = modelName
        self.device = device
        self.batch_size = batch_size
    
        if modelName not in ('BertRost','ESM'):
            print ('Invalid Model Name. Valid Names: "ESM" or "BertRost". Exiting...')
            exit()
        
        if modelName == 'BertRost':
            self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
            self.model = BertForMaskedLM.from_pretrained("Rostlab/prot_bert")
            self.mask = "[MASK]"
            
        if modelName == 'ESM':
            # self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")  #ESM2 this is crap
            # self.model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t12_35M_UR50D") #ESM2 this is crap
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm1b_t33_650M_UR50S")
            self.model = AutoModelForMaskedLM.from_pretrained("facebook/esm1b_t33_650M_UR50S")
            # self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm1v_t33_650M_UR90S_1")
            # self.model = AutoModelForMaskedLM.from_pretrained("facebook/esm1v_t33_650M_UR90S_1")
            self.mask = "<mask>"
        
        try:
            device = int((device))
            # print (device)
        except ValueError:
            print("No CUDA integer given, defaulting to CPU")
            device = 'cpu'
            
        self.unmasker = pipeline('fill-mask', model=self.model, device = self.device, tokenizer=self.tokenizer, top_k = self.topK)


class BertPatcher(bertModels):
    def __init__(self, topK, modelName, device, batch_size):
        super(BertPatcher, self).__init__(topK, modelName, device, batch_size)
        

    def _check_lists(self, listA, listB):
        assert len(listA) == len(listB), 'Error - Lists must be of equal length!'
        
    def _check_string_equality(self, seqA, seqB):
        assert seqA != seqB, 'Some sequences are the same. Exiting...'

    def _make_substitution_index_list(self, seqA, seqB, _id):
        dict_tmp = {i:(resA, resB) for i, (resA, resB) in enumerate(zip(seqA, seqB)) if resA != resB}
        dict_substitutions = {"wt_seq":seqA,
                              "mut_seq":seqB,
                              "substitutions":dict_tmp, 
                              "id": [_id for _ in dict_tmp.keys()]}
        return dict_substitutions

        
    def _make_marginal_sequence_likelihood_masks(self, seqA, seqB):
        pass
    def make_marginal_likelihod_batch(self, listA, listB):
        pass

    def pre_masker(self, reference, target):
    
        all_targets = []
        all_muts = []
        
        for i in range (len(reference)):
            muts = []
            # print (len(reference))
            # print (len(reference[i]))

            assert len(reference[i]) == len(target[i]), 'Error - Sequences must be of equal length!'
            target[i] = list(target[i])
        
            new_ref = reference[i]
            new_target = target[i]
            
            for j, ref in enumerate(new_ref):
                if ref != new_target[j]:
                    muts.append(new_target[j])
                    new_target[j] = 'X'
            new_target = ''.join(new_target)
            all_targets.append(new_target)
            all_muts.append(muts)
           
        return all_targets, all_muts

    def str_prepare(self, orignal_all, mutant_all):
    
        idx_diff = []
        mut_lst = []
        idx_diff_lst = []
      
        for j in range (len(orignal_all)):
            orignal = orignal_all[j]
            mutant = mutant_all[j]
        
            idx_diff = [i for i in range(len(mutant)) if orignal[i] != mutant[i]]
            idx_diff_lst.append(idx_diff)
            replacements = len(idx_diff)*['X']
            mutant = list(mutant)
        
            for (index, replacement) in zip(idx_diff, replacements):
                mutant[index] = replacement
            mutant = ''.join(mutant)
            if self.modelName == 'BertRost':
                mutant = mutant.replace('X', '[MASK]')
                mut_lst.append(mutant)
            else:
                mutant = mutant.replace('X', '<mask>') # Masking style for ESM
                mut_lst.append(mutant)
        
        return mut_lst, idx_diff_lst

    def _str_prepare_single_index(self, dict_substitutions):
    
        idx_diff = list(dict_substitutions["substitutions"])
        wildtype = list(dict_substitutions["wt_seq"])
        mutant = list(dict_substitutions["mut_seq"])
        wt_single_diff_list = []
        mut_single_diff_list = []
        for idx in idx_diff:
            tmp_wildtype = wildtype.copy()
            tmp_wildtype[idx] = self.mask
            wt_single_diff_list.append(" ".join(tmp_wildtype))
            tmp_mutant = mutant.copy()
            tmp_mutant[idx] = self.mask
            mut_single_diff_list.append(" ".join(tmp_mutant))
        dict_substitutions["single_sub_masked_wt"] = wt_single_diff_list
        dict_substitutions["single_sub_masked_mut"] = wt_single_diff_list
        return dict_substitutions

    def top_score_reconstruct(self, top_score, idx_diff, mutant):
        mutant = list(mutant)
        for (index, replacement) in zip(idx_diff, top_score):
            mutant[index] = replacement

        mutant = ''.join(mutant)
        return mutant

    def itter_dict(self, dct, idx, mut):
        mut_score = 0
        new_dct = {}

        # print (f'\nTop Substitutes for AA @ position {idx} (Mutation = {mut})')
        new_dct[idx] = {}
        for i in range (len(dct)):
            # print (f'Substitute AA: {dct[i].get("token_str")} with Score: {dct[i].get("score")}')
            new_dct[idx][dct[i].get("token_str")] = dct[i].get("score")
            if mut == dct[i].get("token_str"):
                mut_score = dct[i].get("score")

        max_like_score = dct[0].get("score")
        return  max_like_score, mut_score, new_dct


    def calc_single_substitution_likelihoods(self, list_wt, list_mut, list_id):

        dict_substitutions_list = list(map(self._make_substitution_index_list, list_wt, list_mut, list_id))
        dict_substitutions_dict = {rec["id"][0]:rec for rec in dict_substitutions_list}
        list_ids = []
        list_ids += [dict_substitutions_dict[_id]["id"] for _id in list_id]
        
        list_dict_substitutions = map(self._str_prepare_single_index, dict_substitutions_list)

        list_single_masked_seq_wt = []
        list_single_masked_seq_mut = []
        for lis in list_dict_substitutions:
            list_single_masked_seq_wt += lis["single_sub_masked_wt"]
            list_single_masked_seq_mut += lis["single_sub_masked_mut"]
        
        ans_all_mut = self.unmasker(list_single_masked_seq_mut, batch_size = self.batch_size)
        ans_all_wt = self.unmasker(list_single_masked_seq_wt, batch_size = self.batch_size)
        dict_ans = {}
        for ans_mut, ans_wt, _id in zip(ans_all_mut, ans_all_wt, list_ids):
            print("flag")
            tmp_mut ={}
            tmp_wt  ={}
            print(_id)
            print(dict_substitutions_dict[_id])
            for ans_var_mut, ans_wt, key in zip(ans_mut, ans_wt, list(dict_substitutions_dict[_id]["substitutions"].keys())):
                tmp_mut[key] = {rec['token_str']:rec["score"] for rec in ans_var_mut}
                tmp_wt[key] = {rec['token_str']:rec["score"] for rec in ans_var_wt}
            dict_ans[d["id"]]= {"WT":tmp_wt,
                               "MUT":tmp_mut}
        return dict_ans
        
    def __call__(self, seqA, seqB):
    
        all_targets, all_muts = self.pre_masker(seqA, seqB)

        mutant_lst, idx_diff_lst = self.str_prepare(seqA, all_targets)
        print(mutant_lst)
        print(idx_diff_lst)
        print ('done')
        
        # for vera A40 512 batch size seems ideal
        ans_all = self.unmasker(mutant_lst, batch_size = self.batch_size)
        
        dct_lst = []
        all_recon = []
        for k in range (len(ans_all)):
            ans = ans_all[k]
            idx_diff = idx_diff_lst[k]
            muts = all_muts[k]
            target = all_targets[k]
            
            top_score = []
            divider = len(ans)
            dict_all = {}
            sum1 = 0
            sum2 = 0
            
            for i in range (len(ans)):
                try:
                    top_score.append(ans[i][0].get('token_str'))
                    score, mut_score, new_dct = self.itter_dict(ans[i], idx_diff[i], muts[i])
                    dict_all.update(new_dct)
                    sum1 += score
                    sum2 += mut_score
                except: # Single Mutation 
                    top_score.append(ans[i].get('token_str'))
                    sum1, sum2, new_dct = self.itter_dict(ans, idx_diff[i], muts[i])
                    dict_all.update(new_dct)
                    divider = 1
                    break
                # print('---------------')
            print (f'Top Substitutions suggested by BERT: {top_score}')
            print (f'Top Substitutions overall likelihood: {sum1/divider}')
            print (f'Overall likelihood of the original substitutions: {sum2/divider}')
            dct_lst.append(dict_all)
            all_recon.append(self.top_score_reconstruct(top_score, idx_diff, target))
        
        return dct_lst, all_recon