# Development Policy

## 1. 目的

この文書は、Himafy の実装品質と保守性を継続的に高めるための開発ポリシーを定義する。

## 2. コーディング規約

- 言語は TypeScript を標準とし、`any` の利用は最小限にする。
- React コンポーネントは関数コンポーネントで実装する。
- 命名規則:
  - 変数・関数: camelCase
  - 型・コンポーネント: PascalCase
  - 定数: UPPER_SNAKE_CASE（共有定数のみ）
- 単一責任を原則とし、1 ファイルで複数の責務を持たせない。
- API レスポンス形式は原則 `{ data, error }` を維持する。
- バリデーションは Zod を利用し、入出力境界で必ず実施する。
- DB スキーマ変更は `supabase/migrations` に追加し、冪等性を意識した SQL を記述する。

## 3. フロントエンド実装方針

- モバイルファーストで設計し、主要画面は 360px 幅で破綻しないことを確認する。
- UI 文言は日本語を基本とする。
- アクセシビリティ要件:
  - 主要インタラクション要素にはキーボード操作性を担保する。
  - フォーム要素にはラベルを付与する。
  - コントラスト不足を避ける。

## 4. ブランチ戦略

- `main` への直接コミットは禁止。
- 作業単位ごとにトピックブランチを作成する。
- ブランチ命名形式:
  - `<type>/<short-description>#<issue-number>`
  - 例: `feat/add-schedule-filter#123`
- `type` は以下から選択する: `feat`, `fix`, `refactor`, `docs`, `chore`, `style`, `test`。
- 長期間ブランチは避け、必要に応じて `main` を取り込んで乖離を減らす。

## 5. コミットメッセージ規約

- Conventional Commits を採用する。
- 基本形式: `<type>(<scope>): <summary>`
- 例:
  - `feat(schedule): add empty-state card`
  - `fix(api): handle missing auth token`
- 1 コミット 1 意図を原則とする。
- 破壊的変更は本文またはフッターで明示する。

## 6. ドキュメント更新規約

- 実装変更により仕様が変わる場合、関連ドキュメントを同 PR で更新する。
- 重要な設計判断は README または docs 配下に理由を残す。
- 運用ルール変更時は `.agent/rules` の関連文書も更新する。

## 7. セキュリティと機密情報

- 秘密情報はコミットしない。
- 環境変数は `.env.local` で管理し、共有時は `.env.local.example` を更新する。
- 外部 API 連携時は、失敗時のフォールバックとログ方針を実装に含める。
