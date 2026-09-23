"""05-5: one final test evaluation after data and decoding are fixed."""
import hashlib
import json
import os
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
os.environ['HF_HOME'] = str(PROJECT / '.cache' / 'huggingface')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import evaluate
import pandas as pd
import torch
from transformers import AutoTokenizer, BartForConditionalGeneration


def json_ready(values):
    return {key: value.item() if hasattr(value, 'item') else value for key, value in values.items()}


def run():
    decoding_dir = PROJECT / 'results/05-4_decoding_validation'
    output = PROJECT / 'results/05-5_final_test_evaluation'
    selected = json.loads((decoding_dir / 'selected_decoding.json').read_text(encoding='utf-8'))
    assert selected['experiment'] == 'A4'
    assert selected['strategy'] == 'beam_5'
    assert selected['test_used'] is False
    checkpoint = Path(selected['checkpoint'])
    test = pd.read_csv(PROJECT / 'processed_data/04-3_final/test.csv', keep_default_na=False)
    assert len(test) == 2564 and test.row_id.is_unique
    test_hash = hashlib.sha256('\n'.join(test.row_id).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    config = {
        'experiment': 'A4', 'checkpoint': str(checkpoint),
        'checkpoint_sha256': hashlib.file_digest(checkpoint.open('rb'), 'sha256').hexdigest(),
        'decoding_strategy': 'beam_5', 'num_beams': 5,
        'test_rows': len(test), 'test_row_id_sha256': test_hash,
        'model_id': 'gogamza/kobart-base-v2', 'max_source_length': 512,
        'max_target_length': 256, 'batch_size': 2,
        'bertscore_model': 'xlm-roberta-base', 'bertscore_rescale_with_baseline': False,
        'test_used_before_this_run': False,
    }
    config_path = output / 'run_config.json'
    if config_path.exists():
        assert json.loads(config_path.read_text(encoding='utf-8')) == config
    else:
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    partial_path = output / 'test_predictions_partial.json'
    predictions = json.loads(partial_path.read_text(encoding='utf-8')) if partial_path.exists() else []
    assert len(predictions) <= len(test)
    assert torch.cuda.is_available()
    tokenizer = AutoTokenizer.from_pretrained(config['model_id'], local_files_only=True,
                                              clean_up_tokenization_spaces=False)
    model = BartForConditionalGeneration.from_pretrained(config['model_id'], local_files_only=True,
                                                          use_safetensors=True)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    model.to('cuda').eval()
    started = time.perf_counter()
    for start in range(len(predictions), len(test), config['batch_size']):
        batch = tokenizer(test.document.iloc[start:start+config['batch_size']].tolist(),
                          padding='max_length', truncation=True,
                          max_length=config['max_source_length'], return_token_type_ids=False,
                          return_tensors='pt').to('cuda')
        with torch.inference_mode():
            generated = model.generate(**batch, max_length=config['max_target_length'],
                                       num_beams=config['num_beams'], do_sample=False)
        predictions.extend(text.strip() for text in tokenizer.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ))
        if len(predictions) % 100 == 0 or len(predictions) == len(test):
            temp = partial_path.with_suffix('.tmp')
            temp.write_text(json.dumps(predictions, ensure_ascii=False), encoding='utf-8')
            temp.replace(partial_path)
            print(f'generation {len(predictions)}/{len(test)}', flush=True)
    prediction_path = output / 'test_predictions.csv'
    if not prediction_path.exists():
        prediction_frame = test[['row_id', 'title', 'document', 'summary']].copy()
        prediction_frame['prediction'] = predictions
        prediction_frame.to_csv(prediction_path, index=False, encoding='utf-8-sig')
    rouge = evaluate.load('rouge').compute(predictions=predictions, references=test.summary.tolist(),
                                           tokenizer=lambda text: text.split(), use_stemmer=False)
    rouge_scores = {key: value * 100 for key, value in rouge.items()}
    print({'rouge': rouge_scores}, flush=True)
    bertscore = evaluate.load('bertscore').compute(
        predictions=predictions, references=test.summary.tolist(), lang='ko',
        model_type=config['bertscore_model'], device='cuda', batch_size=16,
        rescale_with_baseline=config['bertscore_rescale_with_baseline'],
    )
    bert_scores = {
        'bertscore_precision': sum(bertscore['precision']) / len(predictions) * 100,
        'bertscore_recall': sum(bertscore['recall']) / len(predictions) * 100,
        'bertscore_f1': sum(bertscore['f1']) / len(predictions) * 100,
    }
    metrics = json_ready({
        'experiment': 'A4', 'decoding_strategy': 'beam_5', 'num_beams': 5,
        'test_rows': len(test), 'generation_seconds': time.perf_counter() - started,
        **rouge_scores, **bert_scores,
    })
    (output / 'test_metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame([metrics]).to_csv(output / 'test_metrics.csv', index=False, encoding='utf-8-sig')
    print(metrics, flush=True)


if __name__ == '__main__':
    run()
