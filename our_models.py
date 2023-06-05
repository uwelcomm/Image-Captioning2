
from models.blip import blip_decoder
import pytorch_lightning as pl
import json

class TrainingModel(pl.LightningModule):
    def __init__(self,config=None):
        super().__init__()
        if config==None:
            config=json.load(open('config.json','r'))
        if config['load_checkpoint'] and config['checkpoint']:
            self.model = blip_decoder(pretrained=config['checkpoint']+'captioner_ckpt/captioner.pth', image_size=384, vit='base')
        else:
            self.model = blip_decoder(pretrained=config['pretrained'], image_size=384, vit='base')
        self.result = []

        self.config = config
        for param in self.model.visual_encoder.parameters():
            param.requires_grad=config['train_vit']
            

    def training_step(self, batch, batch_idx):
        image, caption,_= batch
        loss = self.model(image, caption)      
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.config['init_lr'], weight_decay=self.config['weight_decay'])
        return optimizer
    
    def validation_step(self, batch, batch_idx):
        image, image_id= batch
        captions = self.model.generate(image, sample=False, num_beams=self.config['num_beams'], max_length=self.config['max_length'], 
                                  min_length=self.config['min_length'])
        for caption, img_id in zip(captions, image_id):
#             result.append({"image_id": img_id.item(), "caption": caption})
            self.result.append({"image_id": img_id, "caption": caption})
    def validation_epoch_end(self,outputs):
        loss=0
        if len(self.result)>=len(self.trainer.val_dataloaders[0])*self.config['batch_size']-self.config['batch_size']:
            val_result_file = save_result(self.result, 'output/Caption_coco/result', 'val_epoch%d'%self.trainer.current_epoch , remove_duplicate='image_id')


            coco_val=coco_caption_eval(self.config['ann_root'],val_result_file,'val')
            loss=coco_val.eval['CIDEr'] + coco_val.eval['Bleu_4']
            self.log_dict({'val_loss': loss,'Bleu_1':coco_val.eval['Bleu_1'],
                                     'Bleu_2':coco_val.eval['Bleu_2'],
                                     'Bleu_3':coco_val.eval['Bleu_3'],
                                     'Bleu_4':coco_val.eval['Bleu_4'],
                                     'CIDEr':coco_val.eval['CIDEr'],
                                     'ROUGE_L':coco_val.eval['ROUGE_L'],
                                     'METEOR':coco_val.eval['METEOR']})


        self.result=[]
        return {"val_loss": loss}
