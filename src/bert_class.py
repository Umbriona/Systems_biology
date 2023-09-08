from transformers import BertForMaskedLM, BertTokenizer, pipeline, AutoTokenizer, AutoModelForMaskedLM


class bertModels:
    def __init__(self, topK, modelName):
        self.topK = topK
        self.modelName = modelName
        if modelName not in ('BertRost','ESM'):
            print ('Invalid Model Name. Valid Names: "ESM" or "BertRost". Exiting...')
            exit()

        if modelName == 'BertRost':
            self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
            self.model = BertForMaskedLM.from_pretrained("Rostlab/prot_bert")

        if modelName == 'ESM':
           # self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")  #ESM2 this is crap
           # self.model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D") #ESM2 this is crap
            #self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm1b_t33_650M_UR50S")
            #self.model = AutoModelForMaskedLM.from_pretrained("facebook/esm1b_t33_650M_UR50S")
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm1v_t33_650M_UR90S_1")
            self.model = AutoModelForMaskedLM.from_pretrained("facebook/esm1v_t33_650M_UR90S_1")

        self.unmasker = pipeline('fill-mask', model=self.model, tokenizer=self.tokenizer, top_k = self.topK, device=1)


class BertPatcher(bertModels):
    def __init__(self, topK=20, modelName = 'BertRost'):
        super(BertPatcher, self).__init__(topK, modelName)


    def pre_masker(self, reference, target):
        muts = []
        try:
            assert (len(reference) == len(target))
        except:
            print ('Error - Sequences must be of equal length!')
            exit()

        target = list(target)
        print(f"Type of target: {type(target)}")
        for i, ref in enumerate(reference):
            if ref != target[i]:
            #    print (f'Got {target[i]}, expected {ref}')
                muts.append(target[i])
                target[i] = 'X'

        target = ''.join(target)
        return target, muts

    def str_prepare(self, orignal, mutant):
        idx_diff = [i for i in range(len(mutant)) if orignal[i] != mutant[i]]
        replacements = len(idx_diff)*['X']
        mutant = list(mutant)

        for (index, replacement) in zip(idx_diff, replacements):
            mutant[index] = replacement

        mutant = ' '.join(mutant)
        if self.modelName == 'BertRost':
            mutant = mutant.replace('X', '[MASK]')
        mutant = mutant.replace('X', '<mask>') # Masking style for ESM 
        return mutant, idx_diff

    def itter_dict(self, dct, idx, mut):
        mut_score = 0
        new_dct = {}

       # print (f'\nTop Substitutes for AA @ position {idx} (Mutation = {mut})')
        new_dct[idx] = {}
        for i in range (len(dct)):
          #  print (f'Substitute AA: {dct[i].get("token_str")} with Score: {dct[i].get("score")}')
            new_dct[idx][dct[i].get("token_str")] = dct[i].get("score")
            if mut == dct[i].get("token_str"):
                mut_score = dct[i].get("score")

        max_like_score = dct[0].get("score")
        return  max_like_score, mut_score, new_dct

    def predict_likelihood(self, ref, query):
        target, muts = self.pre_masker(ref, query)
        mutant, idx_diff = self.str_prepare(ref, target)
        ans = self.unmasker(mutant)
        top_score = [] # this is top amino acid
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
        #    print('---------------')

        print (f'Top Substitutions suggested by BERT: {top_score}')
        print (f'Top Substitutions overall likelihood: {sum1/divider}')
        print (f'Overall likelihood of the original substitutions: {sum2/divider}')

        return dict_all

        
        
    def make_new_reference(self, old_ref, query, id_, bert_flip=False, use_blosum=False, use_grantham = False, thres = 0):
        new_ref = list(old_ref)
        
        if id_ <0.60:
            print("Scipping this one")
            return "".join(new_ref)
        dict_likliehood = self.predict_likelihood(old_ref, query)
       
        
        for rec in dict_likliehood.items():
            ## get idx of mut
            idx = rec[0]
            
            ## get max likelyhood amino acid
            bert_prediction = max(rec[1], key=rec[1].get)
            if query[idx] == "X" or bert_prediction == "X":
                pass  
            elif query[idx] == bert_prediction or (use_blosum and BLOSUM62[bert_prediction][query[idx]] > thres):
               # print(f"Old ref had: '{new_ref[idx]}' New ref will have: '{query[idx]}'")
                new_ref[idx] = query[idx]
            elif query[idx] == bert_prediction or (use_grantham and calc_gram(bert_prediction, query[idx]) < thres):
               # print(f"Old ref had: '{new_ref[idx]}' New ref will have: '{query[idx]}'")
                new_ref[idx] = query[idx]
            elif bert_flip:
                new_ref[idx] = bert_prediction
            if query[idx] == new_ref[idx]:
                print(f"\t\tIndex is: {idx}  Wildtype: {old_ref[idx]} ThermalGAN: {query[idx]} BERT: {bert_prediction} Accepted: {new_ref[idx]} ")
            elif bert_prediction == new_ref[idx]:
                print(f"\tIndex is: {idx}  Wildtype: {old_ref[idx]} ThermalGAN: {query[idx]} BERT: {bert_prediction} Accepted: {new_ref[idx]} ")
            else:
                print(f"Index is: {idx}  Wildtype: {old_ref[idx]} ThermalGAN: {query[idx]} BERT: {bert_prediction} Accepted: {new_ref[idx]} ")
                
        return "".join(new_ref)

    def __call__(self):


        return 0