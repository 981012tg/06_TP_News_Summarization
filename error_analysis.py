"""05-6: final test error analysis and four figures."""
from pathlib import Path
import json
import random

import matplotlib.pyplot as plt
import pandas as pd
from rouge_score.rouge_scorer import RougeScorer

PROJECT = Path(__file__).resolve().parent
INPUT = PROJECT / 'results/05-5_final_test_evaluation'
OUTPUT = PROJECT / 'results/05-6_error_analysis'


def words(text):
    return len(str(text).split())


class WhitespaceTokenizer:
    def tokenize(self, text):
        return str(text).split()


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    test_metrics = json.loads((INPUT / 'test_metrics.json').read_text(encoding='utf-8'))
    frame = pd.read_csv(INPUT / 'test_predictions.csv', keep_default_na=False)
    metadata = pd.read_csv(PROJECT / 'processed_data/04-3_final/test.csv', usecols=['row_id', 'category'])
    frame = frame.merge(metadata, on='row_id', how='left', validate='one_to_one')
    assert len(frame) == 2564 and frame.row_id.is_unique
    assert {'summary', 'prediction', 'category'}.issubset(frame.columns)
    assert frame.category.notna().all()
    scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False,
                         tokenizer=WhitespaceTokenizer())
    row_scores = []
    for reference, prediction in zip(frame.summary, frame.prediction):
        score = scorer.score(reference, prediction)
        row_scores.append({name: score[name].fmeasure * 100 for name in score})
    scores = pd.DataFrame(row_scores)
    result = pd.concat([frame, scores], axis=1)
    result['reference_words'] = result.summary.map(words)
    result['prediction_words'] = result.prediction.map(words)
    result['prediction_reference_ratio'] = result.prediction_words / result.reference_words.clip(lower=1)
    result['length_bin'] = pd.cut(
        result.document.map(words), bins=[0, 100, 250, 500, 1000, float('inf')],
        labels=['≤100', '101–250', '251–500', '501–1000', '>1000'], include_lowest=True,
    )
    result.to_csv(OUTPUT / 'test_row_scores.csv', index=False, encoding='utf-8-sig')
    category = result.groupby('category', observed=True).agg(
        rows=('row_id', 'size'), rouge1=('rouge1', 'mean'), rouge2=('rouge2', 'mean'),
        rougeL=('rougeL', 'mean'), reference_words=('reference_words', 'mean'),
        prediction_words=('prediction_words', 'mean'),
    ).reset_index()
    category.to_csv(OUTPUT / 'category_metrics.csv', index=False, encoding='utf-8-sig')
    lengths = result.groupby('length_bin', observed=True).agg(
        rows=('row_id', 'size'), rougeL=('rougeL', 'mean'),
        reference_words=('reference_words', 'mean'), prediction_words=('prediction_words', 'mean'),
    ).reset_index()
    lengths.to_csv(OUTPUT / 'length_bin_metrics.csv', index=False, encoding='utf-8-sig')
    low = result.nsmallest(50, 'rougeL')
    low.to_csv(OUTPUT / 'low_rougeL_50.csv', index=False, encoding='utf-8-sig')
    result.sample(50, random_state=42).to_csv(OUTPUT / 'random_50.csv', index=False, encoding='utf-8-sig')
    summary = {
        'test_metrics': test_metrics,
        'rows': len(result), 'low_rougeL_threshold_50th': float(low.rougeL.max()),
        'mean_row_rougeL': float(result.rougeL.mean()),
        'median_row_rougeL': float(result.rougeL.median()),
        'analysis_rule': 'per-row ROUGE uses whitespace-tokenized F1; corpus ROUGE remains the official final metric',
    }
    (OUTPUT / 'analysis_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    figure_dir = OUTPUT / 'figures'
    figure_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(result.rougeL, bins=30, color='#3b73af', edgecolor='white')
    ax.axvline(result.rougeL.mean(), color='#d95f02', linestyle='--', label=f'평균 {result.rougeL.mean():.1f}')
    ax.set(xlabel='기사별 ROUGE-L F1', ylabel='기사 수', title='Test 기사별 ROUGE-L 분포')
    ax.legend(); fig.tight_layout(); fig.savefig(figure_dir / '01_rougeL_distribution.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(category.category, category.rougeL, color=['#3b73af', '#77aadd'][:len(category)])
    ax.set_ylim(0, max(category.rougeL.max() + 5, 50))
    ax.set(xlabel='카테고리', ylabel='평균 ROUGE-L F1', title='카테고리별 요약 성능')
    for index, value in enumerate(category.rougeL): ax.text(index, value + .4, f'{value:.1f}', ha='center')
    fig.tight_layout(); fig.savefig(figure_dir / '02_category_rougeL.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(lengths.length_bin.astype(str), lengths.rougeL, color='#4daf4a')
    ax.set(xlabel='본문 길이(어절)', ylabel='평균 ROUGE-L F1', title='본문 길이 구간별 요약 성능')
    ax.tick_params(axis='x', rotation=20)
    fig.tight_layout(); fig.savefig(figure_dir / '03_length_bin_rougeL.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(result.reference_words, result.prediction_words, s=8, alpha=.25, color='#984ea3')
    upper = max(result.reference_words.max(), result.prediction_words.max())
    ax.plot([0, upper], [0, upper], color='#d95f02', linestyle='--', label='동일 길이')
    ax.set(xlabel='정답 요약 길이(어절)', ylabel='생성 요약 길이(어절)', title='생성 요약과 정답 요약의 길이 비교')
    ax.legend(); fig.tight_layout(); fig.savefig(figure_dir / '04_prediction_reference_length.png', dpi=160); plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
