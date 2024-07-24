import yaml

def generate_run_name(config):
    model_params = config['ModelArguments']['model_name_or_path'].split('/')[-1]
    if '8B' in model_params:
        model_size = '8b'
    elif '70B' in model_params:
        model_size = '70b'
    else:
        model_size = 'xb'
    
    model_name = 'llama3_' + model_size
    
    num_train_epochs = config['TrainingArguments']['num_train_epochs']
    visual_features = config['SignDataArguments']['visual_features']
    enabled_features = [key for key, val in visual_features.items() if val['enable_input']]
    visual_features_str = "_".join(enabled_features)
    
    use_paraphrases = '_par' if config['SignDataArguments']['use_paraphrases'] else ''
    label_smoothing = config['TrainingArguments'].get('label_smoothing_factor', 0)
    label_smoothing_str = f"_lsmoothing{int(label_smoothing*10):02}" if label_smoothing > 0 else ''
    
    run_name = f"{model_name}_pre_{num_train_epochs}ep_{visual_features_str}{use_paraphrases}{label_smoothing_str}"
    run_name = run_name.strip('_')
    
    return run_name

file_path = 'pretrain_8B.yaml'

with open(file_path, 'r') as file:
    config = yaml.safe_load(file)

run_name = generate_run_name(config)
print(f"Generated run name: {run_name}")