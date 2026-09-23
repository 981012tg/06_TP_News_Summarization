"""05-4: fixed A4 checkpoint, validation-only decoding comparison."""
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
os.environ['HF_HOME'] = str(PROJECT / '.cache' / 'huggingface')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import hashlib
import json
import time
import numpy as np
import pandas as pd
import torch
import evaluate
from transformers import AutoTokenizer, BartForConditionalGeneration


def run():
    source = PROJECT / 'results' / '05-3_noise_ablation'
    output = PROJECT / 'results' / '05-4_decoding_validation'
    config = json.loads((source / 'run_config.json').read_text(encoding='utf-8'))
    winner = json.loads((source / 'best_validation_experiment.json').read_text(encoding='utf-8'))
    assert winner['experiment'] == 'A4'
    validation = pd.read_csv(PROJECT / 'processed_data/04-3_final/validation.csv')
    digest = hashlib.sha256('\n'.join(validation.row_id).encode()).hexdigest()
    assert digest == config['common_validation_row_id_sha256']
    saved = pd.read_csv(source / 'A4/validation_predictions.csv', keep_default_na=False)
    for column in ['row_id', 'document', 'summary']:
        assert saved[column].tolist() == validation[column].tolist(), column
    assert len(saved) == 2240
    assert torch.cuda.is_available()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = source / 'A4/best_model_state.pt'
    checkpoint_hash = hashlib.file_digest(checkpoint.open('rb'), 'sha256').hexdigest()
    settings = dict(model_id=config['model_id'], model_revision=config['model_revision'],
                    checkpoint_sha256=checkpoint_hash, validation_sha256=digest,
                    max_source_length=512, max_length=256, batch_size=2,
                    strategies=[1, 3, 5], do_sample=False, test_used=False)
    settings_path = output / 'run_config.json'
    if settings_path.exists():
        assert json.loads(settings_path.read_text()) == settings, 'Existing settings differ'
    else:
        settings_path.write_text(json.dumps(settings, indent=2), encoding='utf-8')
    tokenizer = AutoTokenizer.from_pretrained(config['model_id'], revision=config['model_revision'],
                                            local_files_only=True, clean_up_tokenization_spaces=False)
    model = BartForConditionalGeneration.from_pretrained(config['model_id'],
                revision=config['model_revision'], local_files_only=True, use_safetensors=True)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    model.to('cuda').eval()
    metric = evaluate.load('rouge')
    rows = []
    for beams in [1, 3, 5]:
        name = 'greedy' if beams == 1 else f'beam_{beams}'
        partial = output / f'{name}_partial.json'
        predictions = saved.prediction.tolist() if beams == 1 else []
        if beams != 1 and partial.exists():
            predictions = json.loads(partial.read_text(encoding='utf-8'))
        assert len(predictions) <= len(validation)
        started = time.perf_counter()
        for start in range(len(predictions), len(validation), 2):
            batch = tokenizer(validation.document.iloc[start:start+2].tolist(),
                    padding='max_length', truncation=True, max_length=512,
                    return_token_type_ids=False, return_tensors='pt').to('cuda')
            with torch.inference_mode():
                generated = model.generate(**batch, max_length=256, num_beams=beams, do_sample=False)
            predictions.extend(x.strip() for x in tokenizer.batch_decode(generated,
                skip_special_tokens=True, clean_up_tokenization_spaces=False))
            if len(predictions) % 100 == 0 or len(predictions) == len(validation):
                temporary = partial.with_suffix('.tmp')
                temporary.write_text(json.dumps(predictions, ensure_ascii=False), encoding='utf-8')
                temporary.replace(partial)
                print(name, len(predictions), '/', len(validation), flush=True)
        np.random.seed(42)
        scores = metric.compute(predictions=predictions, references=validation.summary.tolist(),
                                tokenizer=lambda text: text.split(), use_stemmer=False)
        row = {'strategy': name, 'num_beams': beams, 'validation_rows': len(validation),
               'seconds_this_run': time.perf_counter()-started,
               **{key: value*100 for key, value in scores.items()}}
        frame = validation[['row_id', 'title', 'document', 'summary']].copy()
        frame['prediction'] = predictions
        frame.to_csv(output / f'{name}_validation_predictions.csv', index=False, encoding='utf-8-sig')
        (output / f'{name}_metrics.json').write_text(json.dumps(row, indent=2), encoding='utf-8')
        rows.append(row)
        print(row, flush=True)
    results = pd.DataFrame(rows).sort_values(['rougeL', 'num_beams'], ascending=[False, True])
    results.to_csv(output / 'decoding_results.csv', index=False, encoding='utf-8-sig')
    selected = {
        key: value.item() if hasattr(value, 'item') else value
        for key, value in results.iloc[0].items()
    }
    selected.update(experiment='A4', checkpoint=str(checkpoint), test_used=False)
    (output / 'selected_decoding.json').write_text(json.dumps(selected, indent=2), encoding='utf-8')
    print('Completed validation decoding comparison', flush=True)


if __name__ == '__main__':
    run()
