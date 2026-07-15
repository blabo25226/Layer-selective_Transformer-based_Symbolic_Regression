# Figures and tables

このディレクトリは、本研究で生成する独立した図・表の正式な保存先である。

## 配置規約

```text
graphs/
  <run-id>/
    figures/   PNG、SVG、PDFなどの図
    tables/    CSV、TSV、TeX、Markdownなどの独立した表
```

- GPU pipelineでは `<run-id>` に `RUN_ID` を使用する。
- CPU pilotを後から可視化するときは `graphs/cpu_pilot/` を使用する。
- ファイル名は `phase4_layer_contribution.svg` のように、Phaseと内容が分かる名前にする。
- 図表の元になったrun ID、JSON、生成スクリプトをcaptionまたは同階層のREADMEに記録する。
- 同じファイル名を上書きせず、異なる実験は別runディレクトリへ保存する。
- 論文掲載用の最終図は可能ならSVGまたはPDFも保存する。

Markdownレポート内のインライン表と、NSRS/TPSRに含まれるvendor資産は移動しない。
独立して再利用・掲載する図表だけをここへ置く。
