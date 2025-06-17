def predict_functional_area(json_path, model_path="fa_classifier.pt", data_path="data"):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

    # Use training dataset to restore label2idx and idx2label
    train_dataset = CrashDataset(data_path, tokenizer)
    label2idx = train_dataset.label2idx
    idx2label = train_dataset.idx2label

    # Load single inference sample
    with open(json_path, 'r') as f:
        sample = json.load(f)

    # Manually process the single sample into model input
    def clean_text(text):
        return text[-4000:] if len(text) > 4000 else text

    crash_type = sample.get('type', '')
    jira_summary = clean_text(sample.get('jira_summary', '') or '')
    inputs = {
        'jira': tokenizer(jira_summary, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
    }

    if crash_type == 'firmware':
        fw_log = clean_text(sample.get('wlan_fw_log', '') or '')
        callstack = clean_text(sample.get('callstack', '') or '')
        inputs['fw_log'] = tokenizer(fw_log, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
        inputs['callstack'] = tokenizer(callstack, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
    else:
        dmesg = clean_text(sample.get('dmesg', '') or '')
        driver = clean_text(sample.get('wlan_driver_log', '') or '')
        inputs['dmesg'] = tokenizer(dmesg, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
        inputs['driver'] = tokenizer(driver, return_tensors='pt', truncation=True, padding='max_length', max_length=512)

    # Load model
    model = FAClassifier(num_classes=len(label2idx))
    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
    # Predict
    with torch.no_grad():
        outputs = model({k: v.to(device) for k, v in inputs.items()})
        pred_idx = outputs.argmax(dim=1).item()
        print(f"Predicted Functional Area: {idx2label[pred_idx]}")
