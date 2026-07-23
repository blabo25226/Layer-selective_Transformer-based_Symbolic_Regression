# Published GPU run summaries

このディレクトリには、`results/runs/<run-id>/`にあるraw GPU成果物から抽出した、
Gitで追跡可能な軽量サマリーを置く。

各`<run-id>/`にはmanifest、validation結果、Phase別集約JSON、Markdown reportを含める。
生成入力データ、checkpoint、巨大ログ、完全なraw runは含めず、研究用ストレージに保存したarchiveを正本とする。
全問題の生の式、簡約式、候補式、変数対応、失敗理由はraw run内に保存されるため、共有ドライブでarchiveも必ず回収する。

生成方法：

```bash
python scripts/validate_gpu_run.py --run-dir results/runs/<run-id>
python scripts/export_run_summary.py --run-dir results/runs/<run-id>
```
