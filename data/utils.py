import re
import json
import os

import torch
import torch.distributed as dist

import utils
from collections import defaultdict
import json,string,re


def load_doc_karpathy(json_file_path,pattern=None,lower=False,is_coco=False):
    # this function return dictionary 
    with open(json_file_path,'r') as file:
        data=json.loads(file.read())
    dict_data=defaultdict(list)
    for example in data['images']:
        temp=[]
        
        if is_coco:
            example['filename']=example['filepath']+'/'+example['filepath']+'/'+example['filename']

        for sentence in example['sentences']:
            cap=sentence['raw']
            if lower:
                cap=cap.lower()
                
            cap=cap.translate(str.maketrans('','',string.punctuation))
            
            if pattern is not None:
                cap=re.sub(pattern,'',cap)
            cap=' '.join(cap.split())

            if example['split']=='train':
                dict_data[example['split']].append({'caption':cap,
                                                    'image': example['filename'],
                                                    'image_id': example['imgid']})
            else:
                temp.append(cap)
                
        if example['split']!='train':
            dict_data[example['split']].append({'caption':temp,
                                                'image': example['filename']
                                             })
            
# {'train':[{'image':'name.jpg','image_id':img_id,'caption':cap}],'test':[{'image':'name.jpg','captions':[cap1,...,cap5]}],'val':[{'image':'name.jpg','captions':[cap1,...,cap5]}]}  
    return dict_data



def pre_caption(caption,max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",       
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption

def prepare_ref_caps_for_evaluate(list_dict,output_json_path=''):
    '''
    list_dict: [{'image':name.jpg,'caption':['cap1','cap2',...]},...]
    returns {'annotations':[{'image_id':name.jpg,'caption':cap,'id':j},...],
                'images':[{'id':name.jpg},...]}
    '''
    
    references={'annotations':[],
                'images':[]}
    j=0
    for i,d in enumerate(list_dict):
        for cap in d['caption']:
            references['annotations'].append({'image_id':d['image'],'caption':cap,'id':j})
            j+=1
        references['images'].append({'id':d['image']})
    if output_json_path:
        json.dump(references,open(output_json_path,'w'))
    
    return references
    
    
def save_result(result, result_dir, filename, remove_duplicate=''):
    result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,utils.get_rank()))
    final_result_file = os.path.join(result_dir, '%s.json'%filename)
    
    json.dump(result,open(result_file,'w'))

#     dist.barrier()

    if utils.is_main_process():   
        # combine results from all processes
        result = []

        for rank in range(utils.get_world_size()):
            result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,rank))
            res = json.load(open(result_file,'r'))
            result += res

        if remove_duplicate:
            result_new = []
            id_list = []    
            for res in result:
                if res[remove_duplicate] not in id_list:
                    id_list.append(res[remove_duplicate])
                    result_new.append(res)
            result = result_new             
                
        json.dump(result,open(final_result_file,'w'))            
        print('result file saved to %s'%final_result_file)

    return final_result_file



from pycocotools.coco import COCO
# from pycocoevalcap.eval import COCOEvalCap
from evaluate.eval import COCOEvalCap
from torchvision.datasets.utils import download_url

def coco_caption_eval(coco_gt_root, results_file, split):
    print()
    print('COMPUTING SCORES FOR '+split+' SPLIT:')
    print()
    urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json',
            'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json'}
    filenames = {'val':'coco_karpathy_val_gt.json','test':'coco_karpathy_test_gt.json'}    
    
#     download_url(urls[split],coco_gt_root)
    annotation_file = os.path.join(coco_gt_root,filenames[split])
    
    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    # for metric, score in coco_eval.eval.items():
    #     print(f'{metric}: {score:.3f}')
    
    return coco_eval
