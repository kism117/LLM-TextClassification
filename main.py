import os
import transformers
from transformers import GPTQConfig
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, Qwen2ForSequenceClassification
from module.argument import ModelArguments,DataArguments,TrainingArguments,LoraArguments

from peft import LoraConfig, get_peft_model
from module.adapter import create_and_replace

import torch
import inspect
import json
from tqdm import tqdm
from sklearn import metrics

from module.auto_from_pretrained import auto_from_pretrained

def train(verbose=False, _val_ = False):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()
    

    device_map = None

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
        is_training=model_args.is_training
    )
    config.use_cache = False
    print(f"checkpoint for config is {config}")

    model = auto_from_pretrained.from_pretrained(model_args.model_name_or_path , num_labels=7)
    model.config.pad_token_id = 151643 
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )


    
    if model_args.add_adapter:
        print("#enter add_adapter")
        create_and_replace(model)
    
    if training_args.use_lora:
        print("#enter lora")
        modules_to_save = ["score",'embed_tokens']

        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
            # task_type="SEQ_CLS",
            modules_to_save=modules_to_save  # This argument serves for adding new tokens.
        )
        model = get_peft_model(model, lora_config)

        # Print peft trainable params
        model.print_trainable_parameters()

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

        for name, param in model.named_parameters():
            # 检查是否是需要设置为可更新的参数
            if name == "base_model.model.score.weight":
                print(f"Setting {name} to be updateable.")
                param.requires_grad = True
            elif name == "base_model.model.model.embed_tokens.weight":
                print(f"Setting {name} to be updateable.")
                param.requires_grad = True
            else:
                pass
    else:
        for name, param in model.model.named_parameters():
            param.requires_grad = False
        

   
    # 检查模型的梯度
    if verbose:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"Parameter Name: {name}, Updateable: True")
            else:
                print(f"Parameter Name: {name}, Updateable: False")
        s = input()
    #return 0
    # exit("exiting......")
    def process_function(examples):
        examples["label"] = [int(unit) for unit in examples["label"]]
        return tokenizer(examples["content"], padding="max_length",truncation=True)

    def load_data(dataset):
        # 加载训练和验证数据集
        dataset = load_dataset("json", data_files={"train": os.path.join(data_args.data_path, "train.json"),
                                                "valid": os.path.join(data_args.data_path, "dev.json")})
        # 使用 map 方法应用数据处理函数，并设置 batched=True 以批量处理数据
        processed_dataset = dataset.map(process_function, batched=True, batch_size=16)
        # 移除不再需要的列，比如 'content' 和 'metadata'
        processed_dataset = processed_dataset.remove_columns(["content", "metadata"])
        return processed_dataset

    processed_data = load_data(data_args)

    
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_data["train"],
        eval_dataset=processed_data["valid"],
    )
    
    # print(training_args.output_dir)
    trainer.train()
    trainer.save_state()
    # trainer.save_model(output_dir=training_args.output_dir)
    # tokenizer.save_pretrained(training_args.output_dir)


    if _val_:
        print("VAL ################################################################### ")
        #model.eval()
        #model.model.eval()
        model.score.eval()

        if verbose:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(f"Parameter Name: {name}, Updateable: True")
                else:
                    print(f"Parameter Name: {name}, Updateable: False")  
            s = input()
        
        y_true = []
        y_pred = []
        with open("data/datasets/longnews/dev.json","r",encoding="utf-8") as f:
            for line in tqdm(f.readlines()):
                example = json.loads(line)
                content = example["content"]
                label = eval(example["label"])
                y_true.append(label)
        
                input_demo = tokenizer(content, padding="max_length",truncation=True,return_tensors="pt")

                for key in input_demo.keys():
                    input_demo[key] = input_demo[key].cuda()

                output = model(**input_demo)

                pred = output.logits.argmax().item()

        
                y_pred.append(pred)

        columns = open("data/datasets/longnews/class.txt","r",encoding="utf-8").readlines()
        columns = [x.strip("\n") for x in columns]

        report = metrics.classification_report(y_true, y_pred, target_names=columns)
        print(report)
    


if __name__ == "__main__":
    train(verbose=False, _val_=True)
